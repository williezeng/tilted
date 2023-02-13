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
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = build_args()
    file_name = os.path.join(constants.YAHOO_DATA_DIR, args['file_name'])
    args["indicators"] = [s.strip() for s in args["indicators"].split(",")]
    data_frame_from_file = tech_indicators.read_df_from_file(file_name)
    if args['model_name'] in NAME_TO_MODEL:
        model_instance = NAME_TO_MODEL[args['model_name']](args, data_frame_from_file)
        model_instance.train_and_predict()
        model_instance.generate_plots()
        analyzer.check_buy_sell_signals(model_instance.ypred, model_instance.ytest)
        analyzer.compute_portvals(model_instance.ypred, data_frame_from_file['Close'])
    else:
        print('must enter a valid model from {}'.format(NAME_TO_MODEL.keys()))
