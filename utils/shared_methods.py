import os
import pandas as pd
import multiprocessing
import joblib
from tqdm import tqdm
from utils import constants
from sklearn.metrics import accuracy_score
import backtrader as bt
from strategy import AlphaStrategy
import pyfolio as pf
import matplotlib.pyplot as plt
from collections import defaultdict


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


def write_prediction_to_csv(predictions_and_file_path):
    prediction_df, prediction_file_path = predictions_and_file_path
    prediction_df.to_csv(prediction_file_path)


def report_generator(cerebro_instance, strat_result, total_trades, stock_name_csv):
    backtesting_report = [f"The following transactions show the backtesting results of {stock_name_csv}'s stock:",
                          f'Starting Portfolio Value: {constants.INITIAL_CAP:,.2f}',
                          f'Final Portfolio Value: {cerebro_instance.broker.getvalue():,.2f}',
                          f'Total number of trades: {total_trades.get("total"):,}',
                          f'Total number of closed trades: {total_trades.get("closed")}',
                          f'Total number of opening trades: {total_trades.get("open")}']

    won = strat_result.analyzers.trade_analysis.get_analysis().get('won')
    if not won:
        return backtesting_report
    backtesting_report.append(
        "Total winning trades: %s (Amount: USD %s)" % (won.get('total'), won.get('pnl').get('total')))
    lost = strat_result.analyzers.trade_analysis.get_analysis().get('lost')
    backtesting_report.append(
        "Total losing trades: %s (Amount: USD %s)" % (lost.get('total'), lost.get('pnl').get('total')))

    pnl = strat_result.analyzers.trade_analysis.get_analysis().get('pnl')
    if not pnl:
        return backtesting_report
    backtesting_report.append(f'Gross profit/loss: USD {pnl.get("gross").get("total"):,.2f}')
    backtesting_report.append(f'Net profit/loss: USD {pnl.get("net").get("total"):,.2f}')
    backtesting_report.append(f'Unrealized gain/loss: USD {cerebro_instance.broker.getvalue() - constants.INITIAL_CAP - pnl.get("net").get("total"):,.2f}')
    backtesting_report.append(f'Total gain/loss: USD {cerebro_instance.broker.getvalue() - constants.INITIAL_CAP:,.2f}')
    backtesting_report.append('')
    backtesting_report.append('Sharpe Ratio: %s' % strat_result.analyzers.mysharpe.get_analysis().get('sharperatio', ''))

    pyfoliozer = strat_result.analyzers.getbyname('pyfolio')

    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    # returns.index = returns.index.tz_convert('UTC')  # Convert to timezone-naive
    returns.name = 'Strategy'
    # backtesting_report.append(pf.show_perf_stats(returns, positions=positions, transactions=transactions, return_df=True))
    # pf.create_position_tear_sheet(returns, positions)
    # plt.savefig('position_tear_sheet.png')
    # plt.close()
    backtesting_report.append('\n' + '-' * 50 + '\n')
    return backtesting_report


def save_predictions_and_accuracy(dir_name):
    # get all test data csv data frames
    # sort based on ticker name
    # remove 'close' price and predict and calculate accuracy
    # save model
    # save accuracy
    results = parallel_get_technical_indicators_and_buy_sell_dfs(get_absolute_file_paths(constants.TESTING_DATA_DIR_PATH))
    predictions_and_file_path = []
    # for a sorted accuracy list
    sorted_results = sorted(results, key=lambda x: x[2])
    suffix_to_yahoo_data_files = defaultdict()
    for x in get_absolute_file_paths(constants.YAHOO_DATA_DIR):
        suffix_to_yahoo_data_files[x.split('/')[-1]] = pd.read_csv(x, index_col=[0], header=[0], skipinitialspace=True, parse_dates=True)
    stock_name_to_portfolio_information = {}
    backtesting_results = []
    for result in tqdm(sorted_results, desc="Predicting and Running Simulation"):
        model = joblib.load(constants.SAVED_MODEL_FILE_PATH)
        reference_technical_indicator_df, reference_buy_sell_df, test_data_file_path = result
        file_name = test_data_file_path.split("/")[-1]
        prefix, suffix = file_name.split('_')
        reference_technical_indicator_df.pop('Close')
        model_predictions = pd.DataFrame(model.predict(reference_technical_indicator_df),
                                         index=reference_technical_indicator_df.index, columns=['bs_signal'])
        if suffix in suffix_to_yahoo_data_files:
            suffix_to_yahoo_data_files[suffix]['openinterest'] = model_predictions
            clean_target_df = suffix_to_yahoo_data_files[suffix].dropna()
        else:
            print(f'suffix {suffix} does not match any filenames in {suffix_to_yahoo_data_files.keys()}')
            continue
        cerebro = bt.Cerebro()
        clean_target_df.pop('Adj Close')
        data = bt.feeds.PandasDirectData(dataname=clean_target_df)
        cerebro.adddata(data)
        cerebro.broker.setcash(constants.INITIAL_CAP)
        cerebro.addstrategy(AlphaStrategy, printlog=False, equity_pct=0.9)
        cerebro.broker.setcommission(commission=0.00)
        # Analyzer
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='draw_down')
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analysis')

        results = cerebro.run()
        strat = results[0]

        total = strat.analyzers.trade_analysis.get_analysis().get('total')
        if total.get('total') == 0:
            continue
        final_portfolio_value = strat.broker.getvalue()
        stock_name_to_portfolio_information[suffix] = {'final_portfolio_value': final_portfolio_value,
                                                       'Percent gain': 100 * (final_portfolio_value - constants.INITIAL_CAP)/constants.INITIAL_CAP,
                                                       'total_trades': total.get('total')}
        backtesting_results.extend(report_generator(cerebro, strat, total, suffix))
        predictions_and_file_path.append((model_predictions, os.path.join(constants.TESTING_PREDICTION_DATA_DIR_PATH, file_name)))
    print('Saving model')

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        __ = pool.map(write_prediction_to_csv, predictions_and_file_path)

    with open(os.path.join(dir_name, constants.BACKTESTING_RESULT_FILE_NAME), 'w') as file:
        file.write('\n'.join(backtesting_results))

    return stock_name_to_portfolio_information


def summary_report(stock_portfolio_information, dir_name):
    sorted_portfolio_info = sorted(stock_portfolio_information.items(),
                                   key=lambda item: item[1]['final_portfolio_value'],
                                   reverse=True)
    report_list = [
        f"Buy/Sell Threshold {constants.BUY_THRESHOLD} within a look ahead day count: {constants.LOOK_AHEAD_DAYS_TO_GENERATE_BUY_SELL}",
        f"Technical Indicators: {constants.TECHNICAL_INDICATORS}",
        f"Model Class Weight: {constants.RANDOM_FOREST_CLASS_WEIGHT}",
        f"{'-' * 50}",
        f"Highest % gain: {sorted_portfolio_info[0]}",
        f"Lowest % gain: {sorted_portfolio_info[-1]}",
        f"Number of stocks with a positive gain {sum(1 for name, info in sorted_portfolio_info if info['final_portfolio_value'] > constants.INITIAL_CAP)}",
        f"Number of stocks with a negative gain {sum(1 for name, info in sorted_portfolio_info if info['final_portfolio_value'] < constants.INITIAL_CAP)}",
        f"Sum of the Top 100 portfolios: {sum(info['final_portfolio_value'] for name, info in sorted_portfolio_info[0:100]):,.2f}",
        f"Sum of the Bottom 100 portfolios: {sum(info['final_portfolio_value'] for name, info in sorted_portfolio_info[-100:]):,.2f}",
        f"{'-' * 50}"
    ]
    print("\n".join(report_list))
    report_list.extend([f"{name}: {info}" for name, info in sorted_portfolio_info])
    with open(os.path.join(dir_name, constants.SUMMARY_REPORT_FILE_NAME), 'w') as file:
        file.write('\n'.join(report_list))
