import os
import argparse
from utils import constants
import tech_indicators
import analyzer
from knn import KNN
from dt import DecisionTree
from logistic_regression import TiltedLogisticRegression
NAME_TO_MODEL = {'knn': KNN,
                 'decision_trees': DecisionTree,
                 'logistic_regression': TiltedLogisticRegression}

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
    parser.add_argument('--file_name', help='the name of the file', type=str, required=True)
    parser.add_argument('--indicators', help='the technical indicators', type=str, required=True)
    parser.add_argument('--optimize_params', help='find best model parameters', type=bool, required=False, default=False)
    parser.add_argument('--model_name', help='the model you want to run', type=str, required=True)
    parser.add_argument('--share_amount', help='the amount of share you want to buy/sell', type=str, required=False, default=100)
    parser.add_argument('--starting_value', help='the starting value', type=str, required=False, default=100000)
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = build_args()
    file_name = os.path.join(constants.YAHOO_DATA_DIR, args['file_name'])
    spy_file_name = os.path.join(constants.YAHOO_DATA_DIR, 'spy500.csv')
    args["indicators"] = [s.strip() for s in args["indicators"].split(",")]
    data_frame_from_file = tech_indicators.read_df_from_file(file_name)
    data_frame_from_spyfile = tech_indicators.read_df_from_file(spy_file_name)
    if args['model_name'] in NAME_TO_MODEL:
        model_instance = NAME_TO_MODEL[args['model_name']](args, data_frame_from_file)
        model_instance.train_and_predict()
        # model_instance.generate_plots()
        analyzer.check_buy_sell_signals(model_instance.ypred, model_instance.ytest)
        long_short_order_book = tech_indicators.add_long_short_shares(model_instance.ypred['bs_signal'], args['share_amount'])
        buy_sell_order_book = tech_indicators.add_buy_sell_shares(model_instance.ypred['bs_signal'], args['share_amount'])
        # order_book.to_csv('tester.csv')
        analyzer.compare_strategies(buy_sell_order_book, long_short_order_book, data_frame_from_file[['Close']], data_frame_from_spyfile[['Close']], args)
        # same buy/long/sell/short signals, just quantity is different
        # analyzer.graph_order_book(buy_sell_portfolio_values, data_frame_from_file[['Close']], args['model_name'], args['file_name'], args["indicators"], args['length'])

    else:
        print('must enter a valid model from {}'.format(NAME_TO_MODEL.keys()))
