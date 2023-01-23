import os
from utils import external_ticks, constants
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import pandas_ta
from talib import BBANDS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tech_indicators
import argparse
import math

# def create_parameter_space(lowest_knn=5):
#     parameter_space = {'penalty': hp.choice('n_neighbors', range(lowest_knn, 70)),
#                        'weights': hp.choice('weights', ['distance']),
#                        'algorithm': hp.choice('algorithm', ['brute', 'ball_tree', 'kd_tree']),
#                        'leaf_size': hp.choice('leaf_size', range(1, 100)),
#                        'p': hp.choice('p', [1, 2]),
#                        'metric': hp.choice('metric', ['minkowski', 'chebyshev']),
#                        }
#     return parameter_space
#
#
# best = math.inf
#
#
# def f(params):
#     global best
#     acc = accuracy_model(params)
#     if acc < best:
#         best = acc
#     return {'loss': acc, 'status': STATUS_OK}
class BaseModel(object):
    def __init__(self, options, data_frame):
        #TODO: DOES this belong here?
        options["indicators"] = [s.strip() for s in options["indicators"].split(",")]
        self.data_frame_from_file = tech_indicators.read_df_from_file(options['name'])
        self.length_of_moving_averages = options.get('length')
        self.indicators = options.get('indicators')
        self.data_name = options.get('data_name')

        self.model = None
        self.model_name = None
        self.rmse = None

    def setup_data(self):
        indicator_df, buy_sell_hold_df = tech_indicators.get_indicators(self.data_frame_from_file, options=self.indicators, length=self.length_of_moving_averages)
        normalized_indicators_df = tech_indicators.normalize_indicators(indicator_df)
        xtrain, xtest, ytrain, ytest = train_test_split(normalized_indicators_df, buy_sell_hold_df, shuffle=False, test_size=0.20)
        return xtrain, xtest, ytrain, ytest

    def train_and_predict(self, xtrain, xtest, ytrain, ytest):
        model = self.model()
        model.fit(xtrain, ytrain)
        self.ypred = model.predict(xtest)  # predicted
        rmse = mean_squared_error(ytest, ypred, squared=False)
        print("Mean Absolute Error: $", rmse)

    def generate_plots(self):
        df2 = pd.DataFrame(data=self.ypred, index=self.ytest.index, columns=['predicted']).astype('float')
        df3 = self.ytest[['bsh_signal']].astype('float').rename(columns={'Close':'actual'})
        ax = df2.plot()
        df3.plot(ax=ax, title='bsh_signal pred values vs real values', fontsize=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('{} bsh values'.format(self.data_name))
        plt.text(0.5, 0.5, 'rmse: '+str(self.rmse), ha='center', va='center', fontsize='small')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{}{}_{}_plot.png".format(self.data_name, self.length_of_moving_averages, self.model_name), dpi=500)



# def build_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
#     parser.add_argument('--name', help='the name of the file', type=str, required=True)
#     parser.add_argument('--indicators', help='the technical indicators', type=str, required=True)
#     return vars(parser.parse_args())
#
# if __name__ == "__main__":
#     user_args = build_args()

#

#     import pdb
#     pdb.set_trace()
#     generate_plots(y_pred, Y_test, rmse, user_args['name'], length_of_moving_averages=user_args['length'])
