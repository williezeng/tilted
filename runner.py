import logging
import os
import argparse
import statistics
import pandas as pd
import analyzer
import multiprocessing
from tqdm import tqdm
import joblib
import graphs
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from utils.constants import TRAINING_DATA_DIR_PATH, TESTING_DATA_DIR_PATH
from tech_indicators import TECHNICAL_INDICATORS, setup_data
from utils import constants, trading_logger, shared_methods
from knn import KNN
from dt import DecisionTree
from rf import RandomForest


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
    live_predictions_average_df = pd.DataFrame()
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
    # The data concatenated and saved is not shuffled for bookkeeping purposes
    # The data is shuffled right before training
    file_name = file_path.split("/")[-1]
    stock_df = pd.read_csv(file_path, index_col=[0], header=[0], skipinitialspace=True)
    stock_df.name = file_name
    try:
        normalized_indicators_df, bs_df, df_for_predictions = setup_data(stock_df, args['indicators'], args['length'], args['lookahead_days'])
        x_train, x_test, y_train, y_test = train_test_split(normalized_indicators_df, bs_df, test_size=0.15, shuffle=False)
        pd.merge(x_train, y_train, left_index=True, right_index=True).to_csv(os.path.join(TRAINING_DATA_DIR_PATH, f'training_{file_name}'))
        pd.merge(x_test, y_test, left_index=True, right_index=True).to_csv(os.path.join(TESTING_DATA_DIR_PATH, f'testing_{file_name}'))
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")


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

    parser.add_argument('--model_name', action='store_true', default=False, help="define the machine learning model")
    parser.add_argument('--all', action='store_true', default=False, help='DO ALL STAGES')
    parser.add_argument('--preprocess_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--combine_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--train_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--predict_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--visualize_all', action='store_true', default=False, help='train a model with the csv files in training_data/')
    parser.add_argument('--skip_training_graphs', help='skip training graphs', action='store_true', default=False)
    parser.add_argument('--skip_testing_graphs', help='skip testing graphs', action='store_true', default=False)

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = build_args()
    # args["indicators"] = [s.strip() for s in args["indicators"].split(",")]
    args["indicators"] = TECHNICAL_INDICATORS
    print(args["indicators"])
    if args['all']:
        args['preprocess_all'] = True
        args['combine_all'] = True
        args['train_all'] = True
        args['predict_all'] = True
        args['visualize_all'] = True
    if args['preprocess_all']:
        print("stage 1: Preprocessing Data")
        list_of_files_in_yahoo_dir = shared_methods.get_absolute_file_paths(constants.YAHOO_DATA_DIR)
        # parallel_data_splitter(list_of_files_in_yahoo_dir[0])
        with multiprocessing.Pool(processes=constants.MULTIPROCESS_CPU_NUMBER) as pool:
            pool.map(parallel_data_splitter, list_of_files_in_yahoo_dir)
        print("stage 1: Preprocessing Data Done")
    if args['combine_all']:
        print("stage 2: Combining Data")
        combine_data(shared_methods.get_absolute_file_paths(constants.TRAINING_DATA_DIR_PATH))
        print("stage 2: Combining Data Done")
    if args['train_all']:
        # shuffle and train on concatenated df
        # Drop 'close' price before training
        # save model
        print("Stage 3: Training Model")
        combined_indicators = pd.read_csv(constants.TRAINING_CONCATENATED_INDICATORS_FILE, index_col='Date', parse_dates=['Date'])
        combined_buy_sell_signals = pd.read_csv(constants.TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE, index_col='Date', parse_dates=['Date'])
        rf = RandomForestClassifier(n_estimators=15, max_depth=30, class_weight=constants.RANDOM_FOREST_CLASS_WEIGHT, n_jobs=-1, random_state=constants.RANDOM_FOREST_RANDOM_STATE)
        # x_train, y_train = shuffle(combined_indicators, combined_buy_sell_signals, random_state=constants.SHUFFLE_RANDOM_STATE)
        combined_indicators.pop('Close')
        rf.fit(combined_indicators, combined_buy_sell_signals['bs_signal'])
        joblib.dump(rf, constants.SAVED_MODEL_FILE_PATH)
        print(f'Training accuracy: {rf.score(combined_indicators, combined_buy_sell_signals)}')
        print("Stage 3: Training Model Done")
    if args['predict_all']:
        # Load and Predict on each Test technical indicator DF
        # Save predictions and compare with correct test buy_sell df
        print("Stage 4: Testing Model")
        shared_methods.save_predictions_and_accuracy()
        print("Stage 4: Testing Model Done")

    if args['visualize_all']:
        print("Visualizing Data")
        if not args['skip_training_graphs']:
            data_map = {'training': shared_methods.get_absolute_file_paths(constants.TRAINING_DATA_DIR_PATH)}
            graphs.visualize_data(data_map)
        if not args['skip_testing_graphs']:
            data_map = {'testing': shared_methods.get_absolute_file_paths(constants.TESTING_DATA_DIR_PATH)}
            graphs.visualize_data(data_map)
        # Todo: guarantee that the file names are equivalently in the same order in the two below data sets
        # Todo: then we can do one index for loop instead of O(n^2)
        data_map = {
            'predictions_buy_sell_files': shared_methods.get_absolute_file_paths(constants.TESTING_PREDICTION_DATA_DIR_PATH),
            'predictions_technical_indicator_files': shared_methods.get_absolute_file_paths(constants.TESTING_DATA_DIR_PATH)
        }
        graphs.visualize_data(data_map)
        print("Visualizing Data Done")
