import logging
import os
import argparse
import pandas as pd
import multiprocessing
import joblib
import shutil
from sklearn.model_selection import train_test_split
from tech_indicators import setup_data
from utils import constants, trading_logger, shared_methods, external_ticks
from datetime import datetime


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




def parallel_data_splitter(tuple_arg):
    # The data concatenated and saved is not shuffled for bookkeeping purposes
    # The data is shuffled right before training
    index, path_to_file, option_args = tuple_arg
    file_name = path_to_file.split("/")[-1]
    stock_df = pd.read_parquet(path_to_file)
    if len(stock_df.index) == 0:
        raise Exception(f'{index}{path_to_file} does not have any data')
    stock_df.name = file_name
    try:
        normalized_indicators_df, bs_df = setup_data(index, stock_df, option_args['indicators'])
        x_train, x_test, y_train, y_test = train_test_split(normalized_indicators_df, bs_df, test_size=0.15, shuffle=False)
        pd.merge(x_train, y_train, left_index=True, right_index=True).to_parquet(os.path.join(constants.TRAINING_DATA_DIR_PATH, f'{file_name}'))
        pd.merge(x_test, y_test, left_index=True, right_index=True).to_parquet(os.path.join(constants.TESTING_DATA_DIR_PATH, f'{file_name}'))
    except Exception as e:
        print(f"Failed to process {index}_{file_name}: {e}")


def combine_data(data_files_map):
    # Read all training technical_indicators and buy sell dfs and concatenate all of them
    technical_indicator_file_path = constants.TRAINING_CONCATENATED_INDICATORS_FILE
    buy_sell_file_path = constants.TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE

    technical_indicators_and_buy_sell_signals = shared_methods.parallel_get_technical_indicators_and_buy_sell_dfs(data_files_map)
    # TODO: ignore_index=True is this necessary?
    all_technical_indicators = pd.concat([technical_indicators_df[0] for technical_indicators_df in technical_indicators_and_buy_sell_signals])
    all_buy_sell_signals = pd.concat([buy_sell_signal_df[1] for buy_sell_signal_df in technical_indicators_and_buy_sell_signals]).to_frame()
    all_technical_indicators.to_parquet(technical_indicator_file_path)
    all_buy_sell_signals.to_parquet(buy_sell_file_path)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', help='the start date of the stock', type=str, default="2008-01-01")
    parser.add_argument('--logger', choices=LOGGER_LEVELS.keys(), default='debug', type=str,
                        help='provide a logging level within {}'.format(LOGGER_LEVELS.keys()))

    parser.add_argument('--gather', action='store_true', default=False, help='gather s&p 500 data from yahoo')
    parser.add_argument('--model_name', type=str, required=True, choices=constants.MODEL_MAP.keys(), help="define the machine learning model")

    parser.add_argument('--tag', required=False, type=str, help='tag this run with a string')
    parser.add_argument('--all', action='store_true', default=False, help='DO ALL STAGES')
    parser.add_argument('--preprocess_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--train_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--predict_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    # Additional check to enforce the conditional requirement
    args = parser.parse_args()
    if not args.gather and not args.model_name:
        parser.error("--model_name is required if --gather is not specified")
    return vars(args)


def create_directory_with_tag(name_of_directory):
    try:
        os.makedirs(name_of_directory)
        print(f"Directory '{name_of_directory}' created successfully.")
    except FileExistsError:
        raise FileExistsError(f"Directory '{name_of_directory}' already exists.")
    except Exception as e:
        raise e


def create_tree(stage):
    creation_map = {'training': [constants.TRAINING_BASE_DIRECTORY_NAME, constants.TRAINING_DATA_DIR_PATH, constants.TRAINING_COMBINED_DATA_DIR_PATH],
                    'testing': [constants.TESTING_BASE_DIRECTORY_NAME, constants.TESTING_DATA_DIR_PATH, constants.TESTING_GRAPHS_DIR_PATH],
                    'predictions': [constants.TESTING_PREDICTION_DIR_PATH, constants.TESTING_PREDICTION_DATA_DIR_PATH, constants.TESTING_PREDICTION_GRAPHS_DIR_PATH]
                    }
    for f_path in creation_map[stage]:
        if os.path.exists(f_path):
            shutil.rmtree(f_path)
        os.makedirs(f_path)


if __name__ == "__main__":
    args = build_args()
    now = datetime.now()
    args["indicators"] = constants.TECHNICAL_INDICATORS
    print(args["indicators"])
    if args['gather']:
        external_ticks.gather_all_fortune500(args['start'], now.strftime("%Y-%m-%d"))
    if args['all']:
        args['preprocess_all'] = True
        args['train_all'] = True
        args['predict_all'] = True
    if args['preprocess_all']:
        print("stage 1: Preprocessing Data")
        # split the training and testing data
        create_tree('training')
        create_tree('testing')
        list_of_files_in_yahoo_dir = [(index, file_path, args) for index, file_path in enumerate(shared_methods.get_absolute_file_paths(constants.YAHOO_DATA_DIR))]
        # parallel_data_splitter(list_of_files_in_yahoo_dir[0])
        with multiprocessing.Pool(processes=constants.MULTIPROCESS_CPU_NUMBER) as pool:
            pool.map(parallel_data_splitter, list_of_files_in_yahoo_dir)
        combine_data(shared_methods.get_absolute_file_paths(constants.TRAINING_DATA_DIR_PATH))
        print("stage 1: Preprocessing Data Done")
    if args['train_all']:
        # shuffle and train on concatenated df
        # Drop 'close' price before training
        # save model
        print("Stage 2: Training Model")
        combined_indicators = pd.read_parquet(constants.TRAINING_CONCATENATED_INDICATORS_FILE)
        combined_buy_sell_signals = pd.read_parquet(constants.TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE)
        ml_model = constants.MODEL_MAP[args['model_name']](**constants.MODEL_ARGS[args['model_name']])
        # x_train, y_train = shuffle(combined_indicators, combined_buy_sell_signals, random_state=constants.SHUFFLE_RANDOM_STATE)
        combined_indicators.pop('Close')
        ml_model.fit(combined_indicators, combined_buy_sell_signals['bs_signal'])
        joblib.dump(ml_model, constants.SAVED_MODEL_FILE_PATH)
        print(f'Training accuracy: {ml_model.score(combined_indicators, combined_buy_sell_signals)}')
        print("Stage 2: Training Model Done")
    if args['predict_all']:
        create_tree('predictions')
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
        predictions_data_structure = shared_methods.market_sim(predictions_structure, directory_name, args['model_name'])
        print("Stage 3: Testing Model Done")
