import os
import argparse
from utils import constants
import tech_indicators
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
    else:
        print('must enter a valid model from {}'.format(NAME_TO_MODEL.keys()))
    # data_frame = tech_indicators.read_df_from_file(name)
    # indicator_dfs, df_close = tech_indicators.get_indicators(data_frame, length=args.length)
    # normalized_indicators = tech_indicators.normalize_indicators(indicator_dfs)
    # # In other words, X_train and X_test is tech indicators
    # # Y_train and Y_test is close price
    # # axis=1 means horizontally concat
    # X = pd.concat(normalized_indicators, axis=1)
    #
    # Y = df_close[['Close']][args.length-1:]
    # Y = Y.astype(float)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.20)
    #
    # y_pred, rmse = train_and_predict(X_train, X_test, Y_train, Y_test, best_parameters)
    # ticker_name = 'Ethereum'
    # generate_plots(y_pred, Y_test, rmse, ticker_name, length_of_moving_averages=args.length)