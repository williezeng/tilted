import matplotlib
import os
import argparse
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
import yfinance as yf
import pyfolio as pf
from utils.constants import BUY, SELL, HOLD
from utils import constants, shared_methods
from datetime import datetime
from tqdm import tqdm


def create_stock_graph(close_data, bs_series, show=False):
    sell_markers_date = [bs_series.index[x] for x in range(len(bs_series)) if bs_series[x] == SELL]
    sell_markers_price = [close_data[x] for x in bs_series.index if bs_series[x] == SELL]
    buy_markers_date = [bs_series.index[x] for x in range(len(bs_series)) if bs_series[x] == BUY]
    buy_markers_price = [close_data[x] for x in bs_series.index if bs_series[x] == BUY]
    plt.plot(list(close_data.index), close_data, label='price')
    plt.plot(sell_markers_date, sell_markers_price, 's', label='sell')
    plt.plot(buy_markers_date, buy_markers_price, 's', label='buy')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
    plt.tight_layout()
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    figure = plt.gcf()
    figure.set_size_inches(20, 10)
    if show:
        plt.show()
    return plt


def create_bar_graph(bs_series):
    buy_counter = 0
    sell_counter = 0
    hold_counter = 0
    for x in range(len(bs_series)):
        if bs_series[x] == SELL:
            sell_counter += 1
        elif bs_series[x] == BUY:
            buy_counter += 1
        else:
            hold_counter += 1
    plt.figure(figsize=(10, 6))
    plt.bar(['buy', 'sell', 'hold'], [buy_counter, sell_counter, hold_counter], color='skyblue')
    plt.title('Buy/Sell/Hold Count')
    plt.xlabel('Actions')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    return plt


def visualize(params, save=True):
    technical_indicators_df, bs_signal_df, title, dir_path = params
    plot_instance = create_stock_graph(technical_indicators_df['Close'], bs_signal_df)
    plot_instance.title(title)
    if save:
        plot_instance.savefig(os.path.join(dir_path, f'{title}_stock.png'), dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.close()

    plot_instance = create_bar_graph(bs_signal_df)
    plot_instance.title(title)
    if save:
        plt.savefig(os.path.join(dir_path, f'{title}_count.png'))
    else:
        plt.show()
    plt.clf()
    plt.close()


def visualize_data(data_files_map):
    params = []
    if len(data_files_map.get('training', [])) > 0:
        print('creating training data graphs')
        for training_file in data_files_map.get('training', []):
            dir_path = constants.TRAINING_GRAPHS_DIR_PATH
            technical_indicators_df, bs_signal_df, __ = shared_methods.get_technical_indicators_and_buy_sell_dfs(training_file)
            title = training_file.split("/")[-1].split(".")[0]
            params.append((technical_indicators_df, bs_signal_df, title, dir_path))
    elif len(data_files_map.get('testing', [])) > 0:
        print('creating testing data graphs')
        for testing_file in data_files_map.get('testing', []):
            dir_path = constants.TESTING_GRAPHS_DIR_PATH
            technical_indicators_df, bs_signal_df, __ = shared_methods.get_technical_indicators_and_buy_sell_dfs(testing_file)
            title = testing_file.split("/")[-1].split(".")[0]
            params.append((technical_indicators_df, bs_signal_df, title, dir_path))
    elif len(data_files_map.get('predictions_buy_sell_files', [])) > 0:
        print('creating prediction graphs')
        for prediction_file in data_files_map.get('predictions_buy_sell_files', []):
            dir_path = constants.TESTING_PREDICTION_GRAPHS_DIR_PATH
            bs_signal_df = pd.read_csv(prediction_file, index_col='Date', parse_dates=['Date'])['bs_signal']
            title = prediction_file.split("/")[-1].split(".")[0]
            for testing_file in data_files_map.get('predictions_technical_indicator_files', []):
                if title == testing_file.split("/")[-1].split(".")[0]:
                    technical_indicators_df, _, __ = shared_methods.get_technical_indicators_and_buy_sell_dfs(testing_file)
                    params.append((technical_indicators_df, bs_signal_df, title, dir_path))
                    break
    else:
        print(f'the map is unexpected {data_files_map}')
        return
    with multiprocessing.Pool(processes=constants.MULTIPROCESS_CPU_NUMBER) as pool:
        results = pool.map(visualize, params)


def read_simulation_df(ticker_report_path):
    stock_close_prices = pd.read_hdf(os.path.join(ticker_report_path, 'stock_close_prices.h5'), 'close')
    portfolio_value_df = pd.read_hdf(os.path.join(ticker_report_path, 'alpha_final_portfolio_value.h5'), 'portfolio_value')
    portfolio_returns = pd.read_hdf(os.path.join(ticker_report_path, 'returns.h5'), 'returns')
    positions = pd.read_hdf(os.path.join(ticker_report_path, 'positions.h5'), 'positions')
    transactions = pd.read_hdf(os.path.join(ticker_report_path, 'transactions.h5'), 'transactions')
    gross_lev = pd.read_hdf(os.path.join(ticker_report_path, 'gross_lev.h5'), 'gross_lev')
    return stock_close_prices, portfolio_value_df, portfolio_returns, positions, transactions, gross_lev


def create_simulation_graphs(portfolio_returns, stock_close_prices, ticker_name, dir_name):
    # compare with benchmark
    benchmark = yf.download('^GSPC', start=stock_close_prices.index[0], end=stock_close_prices.index[-1])['Close']
    benchmark = benchmark.pct_change().dropna().tz_localize('UTC')
    close_prices = stock_close_prices.pct_change().dropna().tz_localize('UTC')
    df = portfolio_returns.to_frame('Strategy').join(benchmark.to_frame('Benchmark (S&P 500)')).join(close_prices.to_frame('Buy and Hold')).dropna()
    fig = plt.figure(figsize=(15, 5))
    df['Strategy'] = (1 + df['Strategy']).cumprod() - 1
    df['Benchmark (S&P 500)'] = (1 + df['Benchmark (S&P 500)']).cumprod() - 1
    df['Buy and Hold'] = (1 + df['Buy and Hold']).cumprod() - 1
    plt.plot(df.index, df['Strategy'], label='Strategy')
    plt.plot(df.index, df['Benchmark (S&P 500)'], label='Benchmark (S&P 500)')
    plt.plot(df.index, df['Buy and Hold'], label='Buy and Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.title(f'{ticker_name} Cumulative Returns: Strategy vs. Benchmark vs Buy and Hold')
    fig.savefig(os.path.join(dir_name, f'{ticker_name}_cumulative_returns.png'))

    fig = plt.figure(figsize=(15, 5))
    pf.plot_annual_returns(portfolio_returns)
    plt.title('Annual Returns of Fund')
    fig.savefig(os.path.join(dir_name, f'{ticker_name}_annual_returns.png'))

    fig = plt.figure(figsize=(15, 5))
    pf.plot_monthly_returns_heatmap(portfolio_returns)
    plt.title('Monthly Returns of Fund (%)')
    fig.savefig(os.path.join(dir_name, f'{ticker_name}_monthly_returns.png'))
    plt.close('all')


def visualize_single_simulation(ticker_report_path, ticker_name):
    stock_close_prices, portfolio_value_df, portfolio_returns, positions, transactions, gross_lev = read_simulation_df(ticker_report_path)
    create_simulation_graphs(portfolio_returns, stock_close_prices, ticker_name, ticker_report_path)


def visualize_all_simulations(capture_path):
    entries = os.listdir(capture_path)
    dirs_in_capture_path = [entry for entry in entries if os.path.isdir(os.path.join(capture_path, entry))]
    for ticker_directory_name in tqdm(dirs_in_capture_path, desc='Creating Graphs'):
        full_path = os.path.join(capture_path, ticker_directory_name)
        stock_close_prices, portfolio_value_df, portfolio_returns, positions, transactions, gross_lev = read_simulation_df(full_path)
        create_simulation_graphs(portfolio_returns, stock_close_prices, ticker_directory_name, full_path)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=False, type=str, help='data_path dir for visualization data')
    parser.add_argument('--ticker_name', required=False, type=str, help='specify a ticker_name for visualization data')

    parser.add_argument('--visualize_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--visualize_single', action='store_true', default=False, help='Visualize a single ticker')

    parser.add_argument('--skip_training_graphs', help='skip training graphs', action='store_true', default=False)
    parser.add_argument('--skip_testing_graphs', help='skip testing graphs', action='store_true', default=False)
    parser.add_argument('--skip_prediction_graphs', help='skip prediction graphs', action='store_true', default=False)
    parser.add_argument('--skip_simulation_graphs', help='skip prediction graphs', action='store_true', default=False)

    args = parser.parse_args()
    if not args.gather and not args.model_name:
        parser.error("--model_name is required if --gather is not specified")
    if args.ticker_name and args.visualize_all:
        parser.error("--ticker_name cannot be used with --visualize_all")
    if args.visualize_all and args.visualize_single:
        parser.error("--visualize_all cannot be used with --visualize_single")
    if args.skip_training_graphs and args.skip_testing_graphs and args.skip_prediction_graphs and args.skip_simulation_graphs:
        parser.error("cannot skip all graphs")
    return vars(args)


if __name__ == "__main__":
    args = build_args()
    now = datetime.now()
    if not args['data_path']:
        raise Exception("args --data_path is required for visualization")
    capture_report_path = os.path.join(constants.PARENT_REPORT_DIRECTORY_NAME, args['data_path'])
    if not os.path.exists(capture_report_path):
        raise Exception("Path does not exist %s. Specify a valid --data_path" % capture_report_path)

    if args['visualize_all']:
        print("Visualizing Data")
        if not args['skip_training_graphs']:
            data_map = {'training': shared_methods.get_absolute_file_paths(constants.TRAINING_DATA_DIR_PATH)}
            visualize_data(data_map)
        if not args['skip_testing_graphs']:
            data_map = {'testing': shared_methods.get_absolute_file_paths(constants.TESTING_DATA_DIR_PATH)}
            visualize_data(data_map)
        # Todo: guarantee that the file names are equivalently in the same order in the two below data sets
        # Todo: then we can do one index for loop instead of O(n^2)
        if not args['skip_prediction_graphs']:
            data_map = {
                'predictions_buy_sell_files': shared_methods.get_absolute_file_paths(constants.TESTING_PREDICTION_DATA_DIR_PATH),
                'predictions_technical_indicator_files': shared_methods.get_absolute_file_paths(constants.TESTING_DATA_DIR_PATH)
            }
            visualize_data(data_map)
        if not args['skip_simulation_graphs']:
            visualize_all_simulations(capture_report_path)
    elif args['visualize_single']:
        if not args['ticker_name']:
            raise Exception("args --ticker_name are required for a single visualization")
        ticker_report_path = os.path.join(capture_report_path, args['ticker_name'])
        if not os.path.exists(ticker_report_path):
            raise Exception("Ticker %s does not exist. Specify a valid --ticker_name" % ticker_report_path)
        visualize_single_simulation(ticker_report_path, args['ticker_name'])
        print("Visualizing Data Done")
