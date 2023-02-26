import logging
import os
import argparse
from utils import constants, trading_logger
import tech_indicators
import analyzer
from knn import KNN
from dt import DecisionTree
from logistic_regression import TiltedLogisticRegression
NAME_TO_MODEL = {'knn': KNN,
                 'decision_trees': DecisionTree,
                 'logistic_regression': TiltedLogisticRegression}
LOGGER_LEVELS = {
                'info': logging.INFO,
                'debug': logging.DEBUG,
                'warning': logging.WARNING,
                'critical': logging.CRITICAL
                 }
logger = trading_logger.getlogger()


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
    parser.add_argument('--logger', choices=LOGGER_LEVELS.keys(), default='debug', type=str, help='provide a logging level within {}'.format(LOGGER_LEVELS.keys()))
    parser.add_argument('--runs', default=1, type=int, help='specify amount of runs')
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = build_args()
    if args['logger'] in LOGGER_LEVELS:
        trading_logger.setlevel(LOGGER_LEVELS[args['logger']])
    else:
        exit('idk your wack ass log level')
    buy_sell_total_percent_gain_runs = []
    long_short_total_percent_gain_runs = []
    test_accuracies_runs = []
    train_accuracies_runs = []

    file_name = os.path.join(constants.YAHOO_DATA_DIR, args['file_name'])
    spy_file_name = os.path.join(constants.YAHOO_DATA_DIR, 'spy500.csv')
    args["indicators"] = [s.strip() for s in args["indicators"].split(",")]
    data_frame_from_ticker = tech_indicators.read_df_from_file(file_name)
    data_frame_from_spyfile = tech_indicators.read_df_from_file(spy_file_name)
    if args['model_name'] not in NAME_TO_MODEL:
        exit('must enter a valid model from {}'.format(NAME_TO_MODEL.keys()))
    for x in range(args["runs"]):
        model_instance = NAME_TO_MODEL[args['model_name']](args, data_frame_from_ticker)
        model_instance.train_and_predict()
        # model_instance.generate_plots()
        analyzer.check_buy_sell_signals(model_instance.ypred, model_instance.ytest)
        long_short_order_book = tech_indicators.add_long_short_shares(model_instance.ypred['bs_signal'], int(args['share_amount']))
        buy_sell_order_book = tech_indicators.add_buy_sell_shares(model_instance.ypred['bs_signal'], data_frame_from_ticker[['Close']],
                                                                  args['starting_value'])
        # order_book.to_csv('tester.csv')
        # tech_indicators.bbands_classification(data_frame_from_ticker)
        buy_sell_portfolio_values, buy_sell_total_percent_gain, long_short_portfolio_values, long_short_total_percent_gain = analyzer.compare_strategies(buy_sell_order_book, long_short_order_book, data_frame_from_ticker[['Close']], data_frame_from_spyfile[['Close']], args)
        # same buy/long/sell/short signals, just quantity is different
        # analyzer.graph_order_book(buy_sell_portfolio_values, data_frame_from_file[['Close']], args['model_name'], args['file_name'], args["indicators"], args['length'])
        # model_instance.save_best_model()
        buy_sell_total_percent_gain_runs.append(buy_sell_total_percent_gain)
        long_short_total_percent_gain_runs.append(long_short_total_percent_gain)
        test_accuracies_runs.append(model_instance.test_score)
        train_accuracies_runs.append(model_instance.train_score)
    output = f""" 
    after {args["runs"]} runs:
    the average percent gain for long shorts is : {sum(long_short_total_percent_gain_runs)/len(long_short_total_percent_gain_runs)}
    the average percent gain for buy sell is : {sum(buy_sell_total_percent_gain_runs)/len(buy_sell_total_percent_gain_runs)}
    the average test accuracy is : {sum(test_accuracies_runs)/len(test_accuracies_runs)}
    the average train accuracy is : {sum(train_accuracies_runs)/len(train_accuracies_runs)}
    """
    print(output)
