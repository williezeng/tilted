import logging
import os
import argparse
import statistics
import pandas
import pandas as pd
from utils import constants, trading_logger
from sklearn.metrics import accuracy_score
import analyzer
from knn import KNN
from dt import DecisionTree
from rf import RandomForest
import multiprocessing
from tqdm import tqdm
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from tech_indicators import TECHNICAL_INDICATORS, setup_data
from sklearn.model_selection import train_test_split
import joblib

from utils.constants import TRAINING_DATA_DIR_PATH, TESTING_DATA_DIR_PATH

NAME_TO_MODEL = {
    'knn': KNN,
    'decision_trees': DecisionTree,
    'random_forest': RandomForest,
}
LOGGER_LEVELS = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warning': logging.WARNING,
    'critical': logging.CRITICAL
}
logger = trading_logger.getlogger()
MIN_REQUIRED_TRADING_INDICATORS = 5


def parallel_trainer_evaluater(tuple_of_data):
    arguments, random_seed, df_from_ticker = tuple_of_data
    arguments['random_seed'] = random_seed
    file_name = arguments['file_name']
    model_name = arguments['model_name']
    indicators = arguments['indicators']
    length = arguments['length']
    share_amount = arguments['share_amount']
    inspect = arguments['inspect']
    save_recent = arguments['save_recent']
    starting_value = arguments['starting_value']
    lookahead_days = arguments['lookahead_days']
    model_instance = NAME_TO_MODEL[arguments['model_name']](arguments, df_from_ticker)
    analyzer.analyze_prediction_to_test(model_instance.ypred, model_instance.ytest, df_from_ticker, share_amount,
                                        starting_value, lookahead_days, save_recent)
    if arguments['check_model']:
        analyzer.graph_series(model_instance.ytrain, df_from_ticker, 'ytrain')
        analyzer.graph_series(model_instance.ytest, df_from_ticker, 'ytest')
        analyzer.graph_series(model_instance.ypred, df_from_ticker, 'ypred')
        model_instance.generate_plots()

    # the simulation starts here:

    buy_sell_portfolio_values, buy_sell_total_percent_gain = analyzer.compare_strategies(
        model_instance.ypred['bs_signal'], df_from_ticker[['Close']], file_name, model_name, indicators, length, share_amount, starting_value, inspect, save_recent)

    live_predictions = model_instance.live_predict()
    return buy_sell_total_percent_gain, model_instance.test_score, model_instance.train_score, live_predictions


def simulation_mode(arguments, data_frame_from_ticker, disable_progress_bar=False):
    results = []
    if arguments['sequential']:
        for run in range(arguments['runs']):
            input_args = (arguments, 6, data_frame_from_ticker)
            results.append(parallel_trainer_evaluater(input_args))
    else:
        with multiprocessing.Pool(processes=12) as pool:
            argument_list = [(arguments, random_seed, data_frame_from_ticker) for random_seed in range(args['runs'])]
            results = []
            with tqdm(total=len(argument_list), disable=disable_progress_bar) as bar:
                for result in pool.imap_unordered(parallel_trainer_evaluater, argument_list):
                    results.append(result)
                    bar.update(1)
    return results


def find_best_combination(arguments, df_ticker):
    # for every combination of tech indicators, spawn threads to train the model defined by the number of runs
    # the training data will be different on every run (thread), but will be the same across combinations (outer scope)
    # This ensures we are comparing each combination against the same training data
    # This ensures we are getting new stats since each run / thread has different training data
    all_combinations = []
    result_buy_sell_total_percent_gain_runs = []
    result_test_accuracies_runs = []
    results_dict = {}
    percent_gain_dict = {}
    for i in range(MIN_REQUIRED_TRADING_INDICATORS, len(arguments["indicators"]) + 1):
        all_combinations += list(combinations(arguments["indicators"], i))
    with tqdm(total=len(all_combinations)) as progress_bar:
        random_seed_from_outer_scope = len(all_combinations)
        for combination in all_combinations:
            arguments['indicators'] = combination
            list_of_results = simulation_mode(arguments, df_ticker, disable_progress_bar=True)
            for buy_sell_percent_gain, test_score, train_score, live_predictions in list_of_results:
                result_buy_sell_total_percent_gain_runs.append(buy_sell_percent_gain)
                result_test_accuracies_runs.append(test_score)
            average_percent_gain = sum(result_buy_sell_total_percent_gain_runs) / len(
                result_buy_sell_total_percent_gain_runs)
            std_percent_gain = statistics.stdev(result_buy_sell_total_percent_gain_runs) if len(
                result_buy_sell_total_percent_gain_runs) > 1 else None
            min_percent_gain = min(result_buy_sell_total_percent_gain_runs)
            max_percent_gain = max(result_buy_sell_total_percent_gain_runs)
            average_test_acc = sum(result_test_accuracies_runs) / len(result_test_accuracies_runs)
            percent_gain_dict = {
                'average_percent_gain': average_percent_gain,
                'std_percent_gain': std_percent_gain,
                'min_percent_gain': min_percent_gain,
                'max_percent_gain': max_percent_gain,
                'average_test_acc': average_test_acc,
            }

            results_dict[tuple(combination)] = percent_gain_dict
            progress_bar.update(1)
    for percent_gain_key in percent_gain_dict:
        count = 0
        print(f'the best to worst {percent_gain_key}')
        if percent_gain_key in ['std_percent_gain']:
            sorted_combinations = sorted(results_dict.items(), key=lambda x: x[1][percent_gain_key], reverse=False)
        else:
            sorted_combinations = sorted(results_dict.items(), key=lambda x: x[1][percent_gain_key], reverse=True)
        for combination, results in sorted_combinations:
            if count == 3:
                break
            print(
                f"Indicators: {sorted(combination)} - {percent_gain_key} - {results[percent_gain_key]}")
            count += 1
        print('*********************')


def run_models(arguments, df_ticker):
    live_predictions_average_df = pandas.DataFrame()
    result_buy_sell_total_percent_gain_runs = []
    # result_long_short_total_percent_gain_runs = []
    result_test_accuracies_runs = []
    result_train_accuracies_runs = []
    list_of_results = simulation_mode(arguments, df_ticker, disable_progress_bar=False)
    for buy_sell_percent_gain, test_score, train_score, live_predictions in list_of_results:
        result_buy_sell_total_percent_gain_runs.append(buy_sell_percent_gain)
        # result_long_short_total_percent_gain_runs.append(long_short_percent_gain)
        result_test_accuracies_runs.append(test_score)
        result_train_accuracies_runs.append(train_score)
        live_predictions_average_df = pd.concat([live_predictions_average_df, live_predictions], axis=1)

    # define a function to count the occurrences of -1, 0, and 1 in a row
    def count_occurrences(row):
        return row.value_counts()

    # apply the function to each row of the dataframe
    counts_df = live_predictions_average_df.apply(count_occurrences, axis=1)

    #      the average percent gain for long shorts is : {sum(result_long_short_total_percent_gain_runs)/len(result_long_short_total_percent_gain_runs)}
    #      the std dev for long shorts is: {statistics.stdev(result_long_short_total_percent_gain_runs)}

    output = f"""
    {df_ticker.name}
    after {arguments["runs"]} runs of the {arguments["model_name"]} with {arguments["length"]} day averages, start value of {arguments["starting_value"]}, share amount of {arguments["share_amount"]}, lookahead days at {arguments['lookahead_days']}, indicators: {sorted(arguments["indicators"])}
    the average percent gain for buy sell is : {sum(result_buy_sell_total_percent_gain_runs) / len(result_buy_sell_total_percent_gain_runs)}
    the std dev total percent gain for buy sell is: {statistics.stdev(result_buy_sell_total_percent_gain_runs) if len(result_buy_sell_total_percent_gain_runs) > 1 else None}
    the min total percent gain for buy sell is: {min(result_buy_sell_total_percent_gain_runs)}
    the max total percent gain for buy sell is: {max(result_buy_sell_total_percent_gain_runs)}
    the average test accuracy is : {sum(result_test_accuracies_runs) / len(result_test_accuracies_runs)}
    the average train accuracy is : {sum(result_train_accuracies_runs) / len(result_train_accuracies_runs)}
    """
    print(output)
    print(counts_df)


def parallel_data_splitter(file_path):
    file_name = file_path.split("/")[-1]
    stock_df = pd.read_csv(file_path, index_col=[0], header=[0], skipinitialspace=True)
    stock_df.name = file_name
    try:
        normalized_indicators_df, bs_df, df_for_predictions = setup_data(stock_df, args['indicators'], args['length'], args['lookahead_days'])
        x_train, x_test, y_train, y_test = train_test_split(normalized_indicators_df, bs_df, test_size=0.15, shuffle=True)
        pd.merge(x_train, y_train, left_index=True, right_index=True).to_csv(os.path.join(TRAINING_DATA_DIR_PATH, f'training_{file_name}'))
        pd.merge(x_test, y_test, left_index=True, right_index=True).to_csv(os.path.join(TESTING_DATA_DIR_PATH, f'testing_{file_name}'))
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")


def get_technical_indcators_and_buy_sell_dfs(file_path):
    technical_indicators_df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
    buy_sell_signal_df = technical_indicators_df.pop('bs_signal')
    return technical_indicators_df, buy_sell_signal_df


def combine_data(data_files_map):
    if len(data_files_map.get('training', [])) > 0:
        indicator_file_path = constants.TRAINING_CONCATENATED_INDICATORS_FILE
        buy_sell_file_path = constants.TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE
        list_of_data_files = data_files_map.get('training', [])
    elif len(data_files_map.get('testing', [])) > 0:
        indicator_file_path = constants.TESTING_CONCATENATED_INDICATORS_FILE
        buy_sell_file_path = constants.TESTING_CONCATENATED_BUY_SELL_SIGNALS_FILE
        list_of_data_files = data_files_map.get('testing', [])
    else:
        print(f'the map is unexpected {data_files_map}')
        return
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(get_technical_indcators_and_buy_sell_dfs, list_of_data_files)
    # TODO: ignore_index=True is this necessary?
    all_technical_indicators = pd.concat([technical_indicators_df[0] for technical_indicators_df in results])
    all_buy_sell_signals = pd.concat([buy_sell_signal_df[1] for buy_sell_signal_df in results])

    all_technical_indicators.to_csv(indicator_file_path)
    all_buy_sell_signals.to_csv(buy_sell_file_path)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
    parser.add_argument('--file_name', help='the name of the file', type=str)
    parser.add_argument('--optimize_params', help='find best model parameters', type=bool, required=False,
                        default=False)
    parser.add_argument('--share_amount', help='the amount of share you want to buy/sell', type=int, required=False,
                        default=10)
    parser.add_argument('--starting_value', help='the starting value', type=int, required=False, default=1000)
    parser.add_argument('--save_recent', help='save the long/short and buy/sell portfolios', type=bool, required=False,
                        default=False)
    parser.add_argument('--inspect', help='inspect the yearly gains', type=bool, required=False, default=False)
    parser.add_argument('--sequential', help='run in sequential', type=bool, required=False, default=False)
    parser.add_argument('--spy', help='only get spy', type=bool, required=False, default=False)
    parser.add_argument('--lookahead_days', help='set the lookahead days for ytest', type=int, required=False,
                        default=6)
    parser.add_argument('--logger', choices=LOGGER_LEVELS.keys(), default='debug', type=str,
                        help='provide a logging level within {}'.format(LOGGER_LEVELS.keys()))
    parser.add_argument('--runs', default=1, type=int, help='specify amount of runs')
    parser.add_argument('--find_best_combination', default=False, type=bool, help='find the best combo of indicators',
                        required=False)
    parser.add_argument('--check_model', type=bool, required=False, default=False,
                        help="specify to see plots of the ytrain ytest ypred generated data")

    parser.add_argument('--model_name', action='store_true', default=False, help="define the machine learning model")
    parser.add_argument('--preprocess_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--combine_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--train_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--test_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--visualize_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    return vars(parser.parse_args())


def get_absolute_file_paths(data_dir):
    file_paths = []

    for filename in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, filename)) and filename.endswith('.csv') and not filename.startswith('00_'):
            file_paths.append(os.path.join(data_dir, filename))
    return file_paths


if __name__ == "__main__":
    args = build_args()
    # args["indicators"] = [s.strip() for s in args["indicators"].split(",")]
    args["indicators"] = TECHNICAL_INDICATORS
    print(args["indicators"])
    if args['preprocess_all']:
        print("stage 1: Preprocessing Data")
        list_of_files_in_yahoo_dir = get_absolute_file_paths(constants.YAHOO_DATA_DIR)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(parallel_data_splitter, list_of_files_in_yahoo_dir)
        print("stage 1: Preprocessing Data Done")
    if args['combine_all']:
        print("stage 2: Combining Data")
        data_map = {'training': get_absolute_file_paths(constants.TRAINING_DATA_DIR_PATH)}
        combine_data(data_map)
        data_map = {'testing': get_absolute_file_paths(constants.TESTING_DATA_DIR_PATH)}
        combine_data(data_map)
        print("stage 2: Combining Data Done")
    if args['train_all']:
        print("stage 3: Training Model")
        x_train = pd.read_csv(constants.TRAINING_CONCATENATED_INDICATORS_FILE, index_col='Date', parse_dates=['Date'])
        y_train = pd.read_csv(constants.TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE, index_col='Date', parse_dates=['Date'])
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf.fit(x_train, y_train['bs_signal'])
        # save
        joblib.dump(rf, constants.SAVED_MODEL_FILE_PATH)
        print(f'Training accuracy: {rf.score(x_train, y_train)}')
        print("Stage 3: Training Model Done")
    if args['predict_all']:
        print("stage 4: Testing Model")
        loaded_rf = joblib.load(constants.SAVED_MODEL_FILE_PATH)
        x_test = pd.read_csv(constants.TESTING_CONCATENATED_INDICATORS_FILE, index_col='Date', parse_dates=['Date'])
        y_test = pd.read_csv(constants.TESTING_CONCATENATED_BUY_SELL_SIGNALS_FILE, index_col='Date', parse_dates=['Date'])

        model_predictions = pd.DataFrame(loaded_rf.predict(x_test), index=x_test.index, columns=['bs_signal'])
        model_predictions.to_csv(constants.PREDICTION_FILE)
        print(f'Testing accuracy: {accuracy_score(y_test, model_predictions)}')
        print("stage 4: Testing Model Done")
    elif args['visualize_all']:
        print("Visualizing Data")
        visualize_data()
        print("Visualizing Data Done")
