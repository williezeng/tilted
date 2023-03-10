import logging
import os
import argparse

import pandas
import pandas as pd
from utils import constants, trading_logger
import tech_indicators
import analyzer
from knn import KNN
from dt import DecisionTree
from rf import RandomForest
import multiprocessing

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


def parallel_trainer_evaluater(arguments, random_seed, df_from_ticker):
    arguments['random_seed'] = random_seed
    model_instance = NAME_TO_MODEL[arguments['model_name']](arguments, df_from_ticker)
    model_instance.train_and_predict()
    # model_instance.generate_plots()
    analyzer.analyze_prediction_to_test(model_instance.ypred, model_instance.ytest, df_from_ticker, arguments)
    long_short_order_book = tech_indicators.add_long_short_shares(model_instance.ypred['bs_signal'],
                                                                  arguments['share_amount'])
    buy_sell_order_book = tech_indicators.add_buy_sell_shares(model_instance.ypred['bs_signal'],
                                                              df_from_ticker[['Close']],
                                                              arguments['starting_value'])
    # model_instance.ypred.to_csv('ypred.csv')
    # long_short_order_book.to_csv('long_short_tester.csv')
    # buy_sell_order_book.to_csv('buy_sell_tester.csv')
    # tech_indicators.bbands_classification(data_frame_from_ticker)
    buy_sell_portfolio_values, buy_sell_total_percent_gain, long_short_portfolio_values, long_short_total_percent_gain = analyzer.compare_strategies(buy_sell_order_book, long_short_order_book, df_from_ticker[['Close']], arguments)
    # same buy/long/sell/short signals, just quantity is different
    # analyzer.graph_order_book(buy_sell_portfolio_values, data_frame_from_file[['Close']], args['model_name'], args['file_name'], args["indicators"], args['length'])
    # model_instance.save_best_model()
    live_predictions = model_instance.live_predict()
    return buy_sell_total_percent_gain, long_short_total_percent_gain, model_instance.test_score, model_instance.train_score, live_predictions


def live_mode(arguments, random_number, df_from_ticker):
    # don't train on all the data, leave some for predicting
    # training_and_testing_data = df_from_ticker.iloc[:-arguments['lookahead_days'], :]

    arguments['random_seed'] = random_number
    model_instance = NAME_TO_MODEL[arguments['model_name']](arguments, df_from_ticker)
    model_instance.train_and_predict()
    analyzer.analyze_prediction_to_test(model_instance.ypred, model_instance.ytest, df_from_ticker, arguments)
    long_short_order_book = tech_indicators.add_long_short_shares(model_instance.ypred['bs_signal'],
                                                                  arguments['share_amount'])
    buy_sell_order_book = tech_indicators.add_buy_sell_shares(model_instance.ypred['bs_signal'],
                                                              df_from_ticker[['Close']],
                                                              arguments['starting_value'])
    buy_sell_portfolio_values, buy_sell_total_percent_gain, long_short_portfolio_values, long_short_total_percent_gain = analyzer.compare_strategies(buy_sell_order_book, long_short_order_book, df_from_ticker[['Close']], arguments)
    live_predictions = model_instance.live_predict()
    return buy_sell_total_percent_gain, long_short_total_percent_gain, model_instance.test_score, model_instance.train_score, live_predictions



def simulation_mode(arguments, data_frame_from_ticker):
    results = []
    if arguments['sequential']:
        for run in range(arguments['runs']):
            results.append(parallel_trainer_evaluater(args, 6, data_frame_from_ticker))
    else:
        with multiprocessing.Pool(processes=4) as pool:
            pool_of_results = pool.starmap_async(parallel_trainer_evaluater, [(arguments, random_seed, data_frame_from_ticker) for random_seed in range(args['runs'])])
            results = pool_of_results.get()
    return results


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
    parser.add_argument('--file_name', help='the name of the file', type=str, required=True)
    parser.add_argument('--indicators', help='the technical indicators', type=str, required=True)
    parser.add_argument('--optimize_params', help='find best model parameters', type=bool, required=False, default=False)
    parser.add_argument('--model_name', help='the model you want to run', type=str, required=True)
    parser.add_argument('--share_amount', help='the amount of share you want to buy/sell', type=int, required=False, default=100)
    parser.add_argument('--starting_value', help='the starting value', type=int, required=False, default=100000)
    parser.add_argument('--save_recent', help='save the long/short and buy/sell portfolios', type=bool, required=False, default=False)
    parser.add_argument('--inspect', help='inspect the yearly gains', type=bool, required=False, default=False)
    parser.add_argument('--sequential', help='run in sequential', type=bool, required=False, default=False)
    parser.add_argument('--spy', help='only get spy', type=bool, required=False, default=False)
    parser.add_argument('--lookahead_days', help='set the lookahead days for ytest', type=int, required=False, default=6)
    parser.add_argument('--logger', choices=LOGGER_LEVELS.keys(), default='debug', type=str, help='provide a logging level within {}'.format(LOGGER_LEVELS.keys()))
    parser.add_argument('--runs', default=1, type=int, help='specify amount of runs')
    parser.add_argument('--live', default=False, type=bool, help='Specify if you want to do live predictions')
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = build_args()
    if args['logger'] not in LOGGER_LEVELS:
        exit('idk your wack ass log level')
    elif args['model_name'] not in NAME_TO_MODEL:
        exit('must enter a valid model from {}'.format(NAME_TO_MODEL.keys()))
    trading_logger.setlevel(LOGGER_LEVELS[args['logger']])
    long_short_buy_sell_tup = []
    file_name = os.path.join(constants.YAHOO_DATA_DIR, args['file_name'])
    spy_file_name = os.path.join(constants.YAHOO_DATA_DIR, 'spy500.csv')
    args["indicators"] = [s.strip() for s in args["indicators"].split(",")]
    data_frame_from_ticker = tech_indicators.read_df_from_file(file_name)
    data_frame_from_spyfile = tech_indicators.read_df_from_file(spy_file_name)
    result_buy_sell_total_percent_gain_runs = []
    result_long_short_total_percent_gain_runs = []
    result_test_accuracies_runs = []
    result_train_accuracies_runs = []
    list_of_results = []
    live_predictions_average_df = pandas.DataFrame()
    if args['spy']:
        analyzer.get_spy(data_frame_from_spyfile, args)
        exit()
    if not args['live']:
        list_of_results = simulation_mode(args, data_frame_from_ticker)
    # else:
    #     # we can parallelize this and get multiple votes
    #     list_of_results = [live_mode(args, 3, data_frame_from_ticker),]

    for buy_sell_percent_gain, long_short_percent_gain, test_score, train_score, live_predictions in list_of_results:
        result_buy_sell_total_percent_gain_runs.append(buy_sell_percent_gain)
        result_long_short_total_percent_gain_runs.append(long_short_percent_gain)
        result_test_accuracies_runs.append(test_score)
        result_train_accuracies_runs.append(train_score)
        live_predictions_average_df = pd.concat([live_predictions_average_df, live_predictions], axis=1)


    # define a function to count the occurrences of -1, 0, and 1 in a row
    def count_occurrences(row):
        return row.value_counts()


    # apply the function to each row of the dataframe
    counts_df = live_predictions_average_df.apply(count_occurrences, axis=1)

    output = f"""
    after {args["runs"]} runs of the {args["model_name"]} with {args["length"]} day averages, start value of {args["starting_value"]}, share amount of {args["share_amount"]}, lookahead days at {args['lookahead_days']}, indicators: {args["indicators"]}
    the average percent gain for long shorts is : {sum(result_long_short_total_percent_gain_runs)/len(result_long_short_total_percent_gain_runs)}
    the average percent gain for buy sell is : {sum(result_buy_sell_total_percent_gain_runs)/len(result_buy_sell_total_percent_gain_runs)}
    the average test accuracy is : {sum(result_test_accuracies_runs)/len(result_test_accuracies_runs)}
    the average train accuracy is : {sum(result_train_accuracies_runs)/len(result_train_accuracies_runs)}
    """
    print(output)
    print(counts_df)
