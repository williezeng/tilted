import os
import pandas as pd
import multiprocessing
import joblib
from tqdm import tqdm
from utils import constants, exceptions
from sklearn.metrics import accuracy_score
import backtrader as bt
from strategy import AlphaStrategy
import pyfolio as pf
import matplotlib.pyplot as plt
from collections import defaultdict
import yfinance as yf
from io import BytesIO


def get_absolute_file_paths(data_dir):
    file_paths = []
    for filename in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, filename)) and filename.endswith('.csv') and not filename.startswith('00_'):
            file_paths.append(os.path.join(data_dir, filename))
    return file_paths


def get_technical_indicators_and_buy_sell_dfs(file_path):
    technical_indicators_df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
    buy_sell_signal_df = technical_indicators_df.pop('bs_signal')
    return technical_indicators_df, buy_sell_signal_df, file_path


def parallel_get_technical_indicators_and_buy_sell_dfs(list_of_data_files):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(get_technical_indicators_and_buy_sell_dfs, list_of_data_files)
    return results


def report_generator(portfolio_result_instance, stock_name):

    total_trades = portfolio_result_instance.analyzers.trade_analysis.get_analysis().get('total')
    final_portfolio_value = portfolio_result_instance.broker.getvalue()

    backtesting_report = [f"The following transactions show the backtesting results of {stock_name}'s stock:",
                          f'Starting Portfolio Value: {constants.INITIAL_CAP:,.2f}',
                          f'Final Portfolio Value: {final_portfolio_value:,.2f}',
                          f'Total number of trades: {total_trades.get("total"):,}',
                          f'Total number of closed trades: {total_trades.get("closed")}',
                          f'Total number of opening trades: {total_trades.get("open")}']

    won = portfolio_result_instance.analyzers.trade_analysis.get_analysis().get('won')
    if not won:
        return backtesting_report
    backtesting_report.append(
        "Total winning trades: %s (Amount: USD %s)" % (won.get('total'), won.get('pnl').get('total')))
    lost = portfolio_result_instance.analyzers.trade_analysis.get_analysis().get('lost')
    backtesting_report.append(
        "Total losing trades: %s (Amount: USD %s)" % (lost.get('total'), lost.get('pnl').get('total')))

    pnl = portfolio_result_instance.analyzers.trade_analysis.get_analysis().get('pnl')
    if not pnl:
        return backtesting_report
    backtesting_report.append(f'Gross profit/loss: USD {pnl.get("gross").get("total"):,.2f}')
    backtesting_report.append(f'Net profit/loss: USD {pnl.get("net").get("total"):,.2f}')
    backtesting_report.append(f'Unrealized gain/loss: USD {final_portfolio_value - constants.INITIAL_CAP - pnl.get("net").get("total"):,.2f}')
    backtesting_report.append(f'Total gain/loss: USD {final_portfolio_value - constants.INITIAL_CAP:,.2f}')
    backtesting_report.append('')
    backtesting_report.append('Sharpe Ratio: %s' % portfolio_result_instance.analyzers.mysharpe.get_analysis().get('sharperatio', ''))


    # backtesting_report.append(pf.show_perf_stats(returns, positions=positions, transactions=transactions, return_df=True))
    # pf.create_position_tear_sheet(returns, positions)
    # plt.savefig('position_tear_sheet.png')
    # plt.close()
    backtesting_report.append('\n' + '-' * 50 + '\n')
    return backtesting_report


def save_predictions():
    # get all test data csv data frames
    # sort based on ticker name
    # remove 'close' price and predict and calculate accuracy
    # save model
    results = parallel_get_technical_indicators_and_buy_sell_dfs(get_absolute_file_paths(constants.TESTING_DATA_DIR_PATH))
    predictions_structure = {}
    ticker_name_to_yahoo_data = defaultdict()
    for file_path in get_absolute_file_paths(constants.YAHOO_DATA_DIR):
        ticker_name_to_yahoo_data[file_path.split('/')[-1].split('.csv')[0]] = pd.read_csv(file_path, index_col=[0], header=[0], skipinitialspace=True, parse_dates=True)
    for result in tqdm(results, desc="Predicting and Running Simulation"):
        model = joblib.load(constants.SAVED_MODEL_FILE_PATH)
        reference_technical_indicator_df, reference_buy_sell_df, test_data_file_path = result
        file_name = test_data_file_path.split("/")[-1]
        ticker_name = file_name.split('.csv')[0]
        stock_close_prices = reference_technical_indicator_df.pop('Close')
        model_predictions = pd.DataFrame(model.predict(reference_technical_indicator_df), index=reference_technical_indicator_df.index, columns=['bs_signal'])
        if ticker_name in ticker_name_to_yahoo_data:
            ticker_name_to_yahoo_data[ticker_name]['openinterest'] = model_predictions
            yahoo_stock_df_with_predictions = ticker_name_to_yahoo_data[ticker_name].dropna()
        else:
            print(f'{ticker_name} does not match any stock names found in {ticker_name_to_yahoo_data.keys()}')
            continue
        if ticker_name in predictions_structure:
            raise Exception(f'A duplicate entry of {ticker_name} has been found. This should be impossible!')
        predictions_structure[ticker_name] = {'stock_df_with_predictions': yahoo_stock_df_with_predictions,
                                              'stock_close_prices': stock_close_prices, 'bs_signal_df': model_predictions}
    return predictions_structure


def write_predictions_to_file(predictions_structure):
    chunks = split_dict(predictions_structure)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        __ = pool.map(process_chunk, chunks)


def process_chunk(chunk):
    # make assertions for predictions_structure
    for ticker_name, prediction_struct in chunk.items():
        prediction_struct['bs_signal_df'].to_csv(os.path.join(constants.TESTING_PREDICTION_DATA_DIR_PATH, ticker_name))


def split_dict(data, num_chunks=5):
    # Get a list of dictionaries
    # Each dictionary has a key len of the chunk_size
    # i.e. if data = 200 entries and num_chunks = 5, each dictionary in the list will have 40 entries
    keys = list(data.keys())
    chunk_size = len(keys) // num_chunks
    return [{k: data[k] for k in keys[i:i + chunk_size]} for i in range(0, len(keys), chunk_size)]


def market_sim(predictions, dir_name):
    stock_name_to_portfolio_information = {}
    backtesting_results = []
    for ticker_name, stock_data_map in predictions.items():
        try:
            simulation_summary_structure = simulator(dir_name, stock_data_map['stock_df_with_predictions'], stock_data_map['stock_close_prices'], ticker_name)
        except exceptions.TradeSimulationException as ex:
            print(ex)
            continue
        backtesting_results.extend(report_generator(simulation_summary_structure['portfolio_result_instance'], ticker_name))
        stock_name_to_portfolio_information[ticker_name] = {'final_portfolio_value': simulation_summary_structure['final_portfolio_value'],
                                                            'Portfolio cumulative percent gain': simulation_summary_structure['Portfolio cumulative percent gain'],
                                                            'Stock percent gain': simulation_summary_structure['Stock percent gain'],
                                                            'total_trades': simulation_summary_structure['total_trades']
                                                            }
    with open(os.path.join(dir_name, constants.BACKTESTING_RESULT_FILE_NAME), 'w') as file:
        file.write('\n'.join(backtesting_results))
    generate_summary_report(stock_name_to_portfolio_information, dir_name)


def visualize_stock_in_simulation(stock_name_to_portfolio_information, dir_name):
    for ticker_name, stock_simulation_map in stock_name_to_portfolio_information.items():
        create_benchmark_graphs(stock_simulation_map['stock_strat_results'], stock_simulation_map['stock_close_prices'], stock_simulation_map['ticker_name'], dir_name)


def simulator(dir_name, clean_target_df, stock_close_prices, ticker_name):
    simulator_path = os.path.join(dir_name, ticker_name)
    os.makedirs(simulator_path, exist_ok=True)
    simulation_summary_data = {}
    alpha_strategy = bt.Cerebro()
    clean_target_df.pop('Adj Close')
    data = bt.feeds.PandasDirectData(dataname=clean_target_df)
    alpha_strategy.adddata(data)
    alpha_strategy.broker.setcash(constants.INITIAL_CAP)
    alpha_strategy.addstrategy(AlphaStrategy, printlog=False, equity_pct=0.9)
    alpha_strategy.broker.setcommission(commission=0.00)
    alpha_strategy.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')
    alpha_strategy.addanalyzer(bt.analyzers.DrawDown, _name='draw_down')
    alpha_strategy.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    alpha_strategy.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analysis')
    results = alpha_strategy.run()
    alpha_strat_results = results[0]
    total = alpha_strat_results.analyzers.trade_analysis.get_analysis().get('total')
    if total.get('total') == 0:
        raise exceptions.TradeSimulationException(f'No trades done for {ticker_name}')

    pyfoliozer = alpha_strat_results.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    returns.index = returns.index.tz_convert('UTC')
    # we need to save as many things from pyfoliozer as possible
    # Assuming alpha_final_portfolio_value is a scalar value
    alpha_final_portfolio_value = alpha_strat_results.broker.getvalue()
    # # Convert the scalar value to a DataFrame
    portfolio_value_df = pd.DataFrame({'final_portfolio_value': [alpha_final_portfolio_value]})
    # # Save the DataFrame to an HDF5 file
    stock_close_prices.to_hdf(os.path.join(simulator_path, 'stock_close_prices.h5'), 'close', mode='w')
    portfolio_value_df.to_hdf(os.path.join(simulator_path, 'alpha_final_portfolio_value.h5'), 'portfolio_value', mode='w')
    returns.to_hdf(os.path.join(simulator_path, 'returns.h5'), 'returns', mode='w')
    positions.to_hdf(os.path.join(simulator_path, 'positions.h5'), 'positions', mode='w')
    transactions.to_hdf(os.path.join(simulator_path, 'transactions.h5'), 'transactions', mode='w')
    gross_lev.to_hdf(os.path.join(simulator_path, 'gross_lev.h5'), 'gross_lev', mode='w')
    simulation_summary_data = {
        'final_portfolio_value': alpha_final_portfolio_value,
        'Portfolio cumulative percent gain': 100 * (
                    alpha_final_portfolio_value - constants.INITIAL_CAP) / constants.INITIAL_CAP,
        'Stock percent gain': 100 * (stock_close_prices[-1] - stock_close_prices[0]) / stock_close_prices[0],
        'total_trades': total.get('total'),
        'portfolio_result_instance': alpha_strat_results,
    }
    return simulation_summary_data


def generate_summary_report(stock_portfolio_information, dir_name):
    sorted_by_portfolio_value = sorted(stock_portfolio_information.items(), key=lambda item: item[1]['final_portfolio_value'], reverse=True)
    sorted_by_portfolio_vs_stock = sorted(stock_portfolio_information.items(), key=lambda item: item[1]['Portfolio cumulative percent gain']-item[1]['Stock percent gain'], reverse=True)
    report_list = [
        f"Buy/Sell Threshold {constants.BUY_THRESHOLD} within a look ahead day count: {constants.LOOK_AHEAD_DAYS_TO_GENERATE_BUY_SELL}",
        f"Technical Indicators: {constants.TECHNICAL_INDICATORS}",
        f"Model Class Weight: {constants.RANDOM_FOREST_CLASS_WEIGHT}",
        f"{'-' * 50}",
        f"Highest % gain: {sorted_by_portfolio_value[0]}",
        f"Lowest % gain: {sorted_by_portfolio_value[-1]}",
        f"Number of stocks with a positive gain {sum(1 for name, info in sorted_by_portfolio_value if info['final_portfolio_value'] > constants.INITIAL_CAP)}",
        f"Number of stocks with a negative gain {sum(1 for name, info in sorted_by_portfolio_value if info['final_portfolio_value'] < constants.INITIAL_CAP)}",
        f"Sum of the Top 100 portfolios: {sum(info['final_portfolio_value'] for name, info in sorted_by_portfolio_value[0:100]):,.2f}",
        f"Sum of the Bottom 100 portfolios: {sum(info['final_portfolio_value'] for name, info in sorted_by_portfolio_value[-100:]):,.2f}",
        f"{'-' * 50}",
    ]
    print("\n".join(report_list))
    report_list.extend([f"Top 100 Portfolios that perform better than their stock"])
    report_list.extend([f"{name}: {info}" for name, info in sorted_by_portfolio_vs_stock[:100]])
    report_list.extend([f"{'-' * 50}", f"All 500 portfolios sorted from Highest portfolio value to least", f"{'-' * 50}"])
    report_list.extend([f"{name}: {info}" for name, info in sorted_by_portfolio_value])
    with open(os.path.join(dir_name, constants.SUMMARY_REPORT_FILE_NAME), 'w') as file:
        file.write('\n'.join(report_list))


# def create_benchmark_graphs(strat_results, stock_close_prices, ticker_name, dir_name):
#     # compare with benchmark
#     benchmark = yf.download('^GSPC', start=stock_close_prices.index[0], end=stock_close_prices.index[-1])['Close']
#     benchmark = benchmark.pct_change().dropna().tz_localize('UTC')
#
#     stock_close_prices = stock_close_prices.pct_change().dropna().tz_localize('UTC')
#
#     df = a_returns.to_frame('Strategy').join(benchmark.to_frame('Benchmark (S&P 500)')).join(stock_close_prices.to_frame('Buy and Hold')).dropna()
#     fig = plt.figure(figsize=(15, 5))
#     df['Strategy'] = (1 + df['Strategy']).cumprod() - 1
#     df['Benchmark (S&P 500)'] = (1 + df['Benchmark (S&P 500)']).cumprod() - 1
#     df['Buy and Hold'] = (1 + df['Buy and Hold']).cumprod() - 1
#     plt.plot(df.index, df['Strategy'], label='Strategy')
#     plt.plot(df.index, df['Benchmark (S&P 500)'], label='Benchmark (S&P 500)')
#     plt.plot(df.index, df['Buy and Hold'], label='Buy and Hold')
#     plt.xlabel('Date')
#     plt.ylabel('Cumulative Returns')
#     plt.legend()
#     plt.grid(True)
#     plt.title(f'{ticker_name} Cumulative Returns: Strategy vs. Benchmark vs Buy and Hold')
#     fig.savefig(os.path.join(dir_name, f'{ticker_name}_cumulative_returns.png'))
#
#     fig = plt.figure(figsize=(15, 5))
#     pf.plot_annual_returns(a_returns)
#     plt.title('Annual Returns of Fund')
#     fig.savefig(os.path.join(dir_name, f'{ticker_name}_annual_returns.png'))
#
#     fig = plt.figure(figsize=(15, 5))
#     pf.plot_monthly_returns_heatmap(a_returns)
#     plt.title('Monthly Returns of Fund (%)')
#     fig.savefig(os.path.join(dir_name, f'{ticker_name}_monthly_returns.png'))
#
#     plt.close('all')
