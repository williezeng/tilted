import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import multiprocessing
from utils.constants import BUY, SELL, HOLD
from utils import constants, shared_methods
import pandas as pd


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
