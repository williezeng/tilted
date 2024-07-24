import logging
import os
import argparse
import pandas as pd
import multiprocessing
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tech_indicators import setup_data
from utils import constants, trading_logger, shared_methods, external_ticks
from rf import RandomForest
from datetime import datetime

NAME_TO_MODEL = {
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


# def find_best_combination(arguments, df_ticker):
#     # for every combination of tech indicators, spawn threads to train the model defined by the number of runs
#     # the training data will be different on every run (thread), but will be the same across combinations (outer scope)
#     # This ensures we are comparing each combination against the same training data
#     # This ensures we are getting new stats since each run / thread has different training data
#     all_combinations = []
#     result_buy_sell_total_percent_gain_runs = []
#     result_test_accuracies_runs = []
#     results_dict = {}
#     percent_gain_dict = {}
#     for i in range(MIN_REQUIRED_TRADING_INDICATORS, len(arguments["indicators"]) + 1):
#         all_combinations += list(combinations(arguments["indicators"], i))
#     with tqdm(total=len(all_combinations)) as progress_bar:
#         random_seed_from_outer_scope = len(all_combinations)
#         for combination in all_combinations:
#             arguments['indicators'] = combination
#             list_of_results = simulation_mode(arguments, df_ticker, disable_progress_bar=True)
#             for buy_sell_percent_gain, test_score, train_score, live_predictions in list_of_results:
#                 result_buy_sell_total_percent_gain_runs.append(buy_sell_percent_gain)
#                 result_test_accuracies_runs.append(test_score)
#             average_percent_gain = sum(result_buy_sell_total_percent_gain_runs) / len(
#                 result_buy_sell_total_percent_gain_runs)
#             std_percent_gain = statistics.stdev(result_buy_sell_total_percent_gain_runs) if len(
#                 result_buy_sell_total_percent_gain_runs) > 1 else None
#             min_percent_gain = min(result_buy_sell_total_percent_gain_runs)
#             max_percent_gain = max(result_buy_sell_total_percent_gain_runs)
#             average_test_acc = sum(result_test_accuracies_runs) / len(result_test_accuracies_runs)
#             percent_gain_dict = {
#                 'average_percent_gain': average_percent_gain,
#                 'std_percent_gain': std_percent_gain,
#                 'min_percent_gain': min_percent_gain,
#                 'max_percent_gain': max_percent_gain,
#                 'average_test_acc': average_test_acc,
#             }
#
#             results_dict[tuple(combination)] = percent_gain_dict
#             progress_bar.update(1)
#     for percent_gain_key in percent_gain_dict:
#         count = 0
#         print(f'the best to worst {percent_gain_key}')
#         if percent_gain_key in ['std_percent_gain']:
#             sorted_combinations = sorted(results_dict.items(), key=lambda x: x[1][percent_gain_key], reverse=False)
#         else:
#             sorted_combinations = sorted(results_dict.items(), key=lambda x: x[1][percent_gain_key], reverse=True)
#         for combination, results in sorted_combinations:
#             if count == 3:
#                 break
#             print(
#                 f"Indicators: {sorted(combination)} - {percent_gain_key} - {results[percent_gain_key]}")
#             count += 1
#         print('*********************')


# def run_models(arguments, df_ticker):
#     live_predictions_average_df = pd.DataFrame()
#     result_buy_sell_total_percent_gain_runs = []
#     # result_long_short_total_percent_gain_runs = []
#     result_test_accuracies_runs = []
#     result_train_accuracies_runs = []
#     list_of_results = simulation_mode(arguments, df_ticker, disable_progress_bar=False)
#     for buy_sell_percent_gain, test_score, train_score, live_predictions in list_of_results:
#         result_buy_sell_total_percent_gain_runs.append(buy_sell_percent_gain)
#         # result_long_short_total_percent_gain_runs.append(long_short_percent_gain)
#         result_test_accuracies_runs.append(test_score)
#         result_train_accuracies_runs.append(train_score)
#         live_predictions_average_df = pd.concat([live_predictions_average_df, live_predictions], axis=1)
#
#     # define a function to count the occurrences of -1, 0, and 1 in a row
#     def count_occurrences(row):
#         return row.value_counts()
#
#     # apply the function to each row of the dataframe
#     counts_df = live_predictions_average_df.apply(count_occurrences, axis=1)
#
#     #      the average percent gain for long shorts is : {sum(result_long_short_total_percent_gain_runs)/len(result_long_short_total_percent_gain_runs)}
#     #      the std dev for long shorts is: {statistics.stdev(result_long_short_total_percent_gain_runs)}
#
#     output = f"""
#     {df_ticker.name}
#     after {arguments["runs"]} runs of the {arguments["model_name"]} with {arguments["length"]} day averages, start value of {arguments["starting_value"]}, share amount of {arguments["share_amount"]}, lookahead days at {arguments['lookahead_days']}, indicators: {sorted(arguments["indicators"])}
#     the average percent gain for buy sell is : {sum(result_buy_sell_total_percent_gain_runs) / len(result_buy_sell_total_percent_gain_runs)}
#     the std dev total percent gain for buy sell is: {statistics.stdev(result_buy_sell_total_percent_gain_runs) if len(result_buy_sell_total_percent_gain_runs) > 1 else None}
#     the min total percent gain for buy sell is: {min(result_buy_sell_total_percent_gain_runs)}
#     the max total percent gain for buy sell is: {max(result_buy_sell_total_percent_gain_runs)}
#     the average test accuracy is : {sum(result_test_accuracies_runs) / len(result_test_accuracies_runs)}
#     the average train accuracy is : {sum(result_train_accuracies_runs) / len(result_train_accuracies_runs)}
#     """
#     print(output)
#     print(counts_df)


def parallel_data_splitter(tuple_arg):
    # The data concatenated and saved is not shuffled for bookkeeping purposes
    # The data is shuffled right before training
    index, path_to_file, option_args = tuple_arg
    file_name = path_to_file.split("/")[-1]
    stock_df = pd.read_csv(path_to_file, index_col=[0], header=[0], skipinitialspace=True)
    if len(stock_df.index) == 0:
        raise Exception(f'{index}{path_to_file} does not have any data')
    stock_df.name = file_name
    try:
        normalized_indicators_df, bs_df, df_for_predictions = setup_data(index, stock_df, option_args['indicators'], option_args['length'], option_args['lookahead_days'])
        x_train, x_test, y_train, y_test = train_test_split(normalized_indicators_df, bs_df, test_size=0.15, shuffle=False)
        pd.merge(x_train, y_train, left_index=True, right_index=True).to_csv(os.path.join(constants.TRAINING_DATA_DIR_PATH, f'{file_name}'))
        pd.merge(x_test, y_test, left_index=True, right_index=True).to_csv(os.path.join(constants.TESTING_DATA_DIR_PATH, f'{file_name}'))
    except Exception as e:
        print(f"Failed to process {index}_{file_name}: {e}")


def combine_data(data_files_map):
    # Read all training technical_indicators and buy sell dfs and concatenate all of them
    technical_indicator_file_path = constants.TRAINING_CONCATENATED_INDICATORS_FILE
    buy_sell_file_path = constants.TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE

    technical_indicators_and_buy_sell_signals = shared_methods.parallel_get_technical_indicators_and_buy_sell_dfs(data_files_map)
    # TODO: ignore_index=True is this necessary?
    all_technical_indicators = pd.concat([technical_indicators_df[0] for technical_indicators_df in technical_indicators_and_buy_sell_signals])
    all_buy_sell_signals = pd.concat([buy_sell_signal_df[1] for buy_sell_signal_df in technical_indicators_and_buy_sell_signals])
    all_technical_indicators.to_csv(technical_indicator_file_path)
    all_buy_sell_signals.to_csv(buy_sell_file_path)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
    parser.add_argument('--file_name', help='the name of the file', type=str)
    parser.add_argument('--optimize_params', help='find best model parameters', type=bool, required=False,
                        default=False)
    parser.add_argument('--share_amount', help='the amount of share you want to buy/sell', type=int, required=False,
                        default=5)
    parser.add_argument('--starting_value', help='the starting value', type=int, required=False, default=1000)
    parser.add_argument('--save_recent', help='save the long/short and buy/sell portfolios', type=bool, required=False,
                        default=False)
    parser.add_argument('--inspect', help='inspect the yearly gains', type=bool, required=False, default=False)
    parser.add_argument('--sequential', help='run in sequential', type=bool, required=False, default=False)
    parser.add_argument('--spy', help='only get spy', type=bool, required=False, default=False)
    parser.add_argument('--lookahead_days', help='set the lookahead days for ytest', type=int, required=False,
                        default=10)
    parser.add_argument('--logger', choices=LOGGER_LEVELS.keys(), default='debug', type=str,
                        help='provide a logging level within {}'.format(LOGGER_LEVELS.keys()))
    parser.add_argument('--runs', default=1, type=int, help='specify amount of runs')
    parser.add_argument('--find_best_combination', default=False, type=bool, help='find the best combo of indicators',
                        required=False)
    parser.add_argument('--check_model', type=bool, required=False, default=False,
                        help="specify to see plots of the ytrain ytest ypred generated data")

    parser.add_argument('--gather', action='store_true', default=False, help='gather s&p 500 data from yahoo')
    parser.add_argument('--model_name', action='store_true', default=False, help="define the machine learning model")

    parser.add_argument('--tag', required=False, type=str, help='tag this run with a string')
    parser.add_argument('--all', action='store_true', default=False, help='DO ALL STAGES')
    parser.add_argument('--preprocess_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--combine_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--train_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--predict_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    return vars(parser.parse_args())


def create_directory_with_tag(name_of_directory):
    try:
        os.makedirs(name_of_directory)
        print(f"Directory '{name_of_directory}' created successfully.")
    except FileExistsError:
        raise FileExistsError(f"Directory '{name_of_directory}' already exists.")
    except Exception as e:
        raise e


if __name__ == "__main__":
    args = build_args()
    now = datetime.now()
    args["indicators"] = constants.TECHNICAL_INDICATORS
    print(args["indicators"])
    if args['gather']:
        external_ticks.gather_all_fortune500()
    if args['all']:
        args['preprocess_all'] = True
        args['train_all'] = True
        args['predict_all'] = True
    if args['preprocess_all']:
        print("stage 1: Preprocessing Data")
        for file_path in [constants.TRAINING_BASE_DIRECTORY_NAME, constants.TRAINING_DATA_DIR_PATH, constants.TRAINING_COMBINED_DATA_DIR_PATH,
                          constants.TESTING_BASE_DIRECTORY_NAME, constants.TESTING_DATA_DIR_PATH, constants.TESTING_GRAPHS_DIR_PATH,
                          constants.TESTING_PREDICTION_DIR_PATH, constants.TESTING_PREDICTION_DATA_DIR_PATH, constants.TESTING_PREDICTION_GRAPHS_DIR_PATH]:
            if not os.path.exists(file_path):
                os.mkdir(file_path)
        list_of_files_in_yahoo_dir = [(index, file_path, args) for index, file_path in enumerate(shared_methods.get_absolute_file_paths(constants.YAHOO_DATA_DIR))]
        with multiprocessing.Pool(processes=constants.MULTIPROCESS_CPU_NUMBER) as pool:
            pool.map(parallel_data_splitter, list_of_files_in_yahoo_dir)
        combine_data(shared_methods.get_absolute_file_paths(constants.TRAINING_DATA_DIR_PATH))
        print("stage 1: Preprocessing Data Done")
    if args['train_all']:
        # shuffle and train on concatenated df
        # Drop 'close' price before training
        # save model
        print("Stage 2: Training Model")
        combined_indicators = pd.read_csv(constants.TRAINING_CONCATENATED_INDICATORS_FILE, index_col='Date', parse_dates=['Date'])
        combined_buy_sell_signals = pd.read_csv(constants.TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE, index_col='Date', parse_dates=['Date'])
        rf = RandomForestClassifier(n_estimators=15, max_depth=25, n_jobs=-1, class_weight=constants.RANDOM_FOREST_CLASS_WEIGHT, random_state=constants.RANDOM_FOREST_RANDOM_STATE)
        # x_train, y_train = shuffle(combined_indicators, combined_buy_sell_signals, random_state=constants.SHUFFLE_RANDOM_STATE)
        combined_indicators.pop('Close')
        rf.fit(combined_indicators, combined_buy_sell_signals['bs_signal'])
        joblib.dump(rf, constants.SAVED_MODEL_FILE_PATH)
        print(f'Training accuracy: {rf.score(combined_indicators, combined_buy_sell_signals)}')
        print("Stage 2: Training Model Done")
    if args['predict_all']:
        # Load and Predict on each Test technical indicator DF
        # Save predictions and compare with correct test buy_sell df
        print("Stage 3: Testing Model")
        # Get current date and time
        short_date_time = now.strftime('%y%m%d_%H%M')
        directory_name = constants.FULL_REPORT_DIR.format(short_date_time, args['tag'])
        create_directory_with_tag(directory_name)
        predictions_structure = shared_methods.save_predictions()
        print('Running simulation and writing predictions')
        shared_methods.write_predictions_to_file(predictions_structure)
        predictions_data_structure = shared_methods.market_sim(predictions_structure, directory_name)
        print("Stage 3: Testing Model Done")
