import logging
import os
import argparse
import statistics
import pandas
import pandas as pd
from utils import constants, trading_logger
import tech_indicators
import analyzer
from knn import KNN
from dt import DecisionTree
from rf import RandomForest
import multiprocessing
from tqdm import tqdm
from itertools import combinations
from utils import external_ticks
import matplotlib.pyplot as plt
from tech_indicators import TECHNICAL_INDICATORS
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


def get_df_from_file(f_path):
    data_frame_from_ticker = tech_indicators.read_df_from_file(f_path)
    data_frame_from_ticker.name = f_path
    return data_frame_from_ticker


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
    parser.add_argument('--file_name', help='the name of the file', type=str)
    parser.add_argument('--indicators',
                        help=f'the technical indicators, you must specify one of{TECHNICAL_INDICATORS}',
                        type=str, required=True)
    parser.add_argument('--optimize_params', help='find best model parameters', type=bool, required=False,
                        default=False)
    parser.add_argument('--model_name', help='the model you want to run', type=str, required=True)
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
    parser.add_argument('--all', type=bool, default=False, help='run through all data inside yahoo_data')
    return vars(parser.parse_args())

from models import BaseModel
if __name__ == "__main__":
    args = build_args()
    if args['logger'] not in LOGGER_LEVELS:
        exit('idk your wack ass log level')
    elif args['model_name'] not in NAME_TO_MODEL:
        exit('must enter a valid model from {}'.format(NAME_TO_MODEL.keys()))
    trading_logger.setlevel(LOGGER_LEVELS[args['logger']])
    args["indicators"] = [s.strip() for s in args["indicators"].split(",")]
    # if len(args["indicators"]) < MIN_REQUIRED_TRADING_INDICATORS:
    #     exit(f"you must specify {MIN_REQUIRED_TRADING_INDICATORS} or more indicators")
    if args['spy']:
        spy_file_name = os.path.join(constants.YAHOO_DATA_DIR, 'SPY.csv')
        data_frame_from_spyfile = tech_indicators.read_df_from_file(spy_file_name)
        analyzer.get_spy(data_frame_from_spyfile, args)
        exit()
    if args['all']:
        files = os.listdir(constants.YAHOO_DATA_DIR)
        # Filter files that do not start with '00-'
        filtered_files = [file for file in files if not file.startswith('00-')]
        for file_name in filtered_files:
            file_path = os.path.join(constants.YAHOO_DATA_DIR, file_name)
            df_from_ticker = get_df_from_file(file_path)
            dt_instance = NAME_TO_MODEL[args['model_name']](args, df_from_ticker)
            normalized_indicators_df, bs_df, df_for_predictions = dt_instance.setup_data()
            normalized_indicators_to_buy_sell_df = pd.merge(normalized_indicators_df, bs_df, left_index=True, right_index=True)
            # split training
            # split testing
            df.to_csv(os.path.join(DATA_DIR, '{}.csv'.format(name)))

    elif args['file_name']:
        file_path = os.path.join(constants.YAHOO_DATA_DIR, args['file_name'])
        df_from_ticker = get_df_from_file(file_path)
        if args['find_best_combination']:
            find_best_combination(args, df_from_ticker)
            exit()
        else:
            run_models(args, df_from_ticker)
    else:
        print(f'not sure what to do given {args}')
