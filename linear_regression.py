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

def train_and_predict(xtrain, xtest, ytrain, ytest):
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)  # predicted
    # Printout relevant metrics
    rmse = mean_squared_error(ytest, y_pred, squared=False)
    print("Model Coefficients:", model.coef_)
    print("Mean Absolute Error: $", mean_absolute_error(ytest, y_pred))
    print("Root Mean Square Error: $", rmse)

    print("Coefficient of Determination:", r2_score(ytest, y_pred))
    regression_confidence = model.score(xtest, ytest)
    print("linear regression confidence: ", regression_confidence)
    return y_pred, rmse

def generate_plots(y_pred, Y_test, rmse, name, length_of_moving_averages=10):
    df2 = pd.DataFrame(data=y_pred, index=Y_test.index, columns=['predicted']).astype('float')
    import pdb
    pdb.set_trace()
    df3 = Y_test[['Close']].astype('float').rename(columns={'Close':'actual'})
    ax = df2.plot()
    df3.plot(ax=ax, title='pred values vs real values', fontsize=10)

    ax.set_xlabel('Date')
    ax.set_ylabel('{} Price'.format(name))
    plt.text(0.5, 0.5, 'rmse: '+str(rmse), ha='center', va='center', fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("{}{}_decision_tree_plot.png".format(name, length_of_moving_averages), dpi=500)

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowest_knn_neighbor', help='we need to separate search and knn', type=int)
    parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
    parser.add_argument('--name', help='the name of the file', type=str, required=True)
    parser.add_argument('--indicators', help='the technical indicators', type=str, required=True)
    return vars(parser.parse_args())

if __name__ == "__main__":
    user_args = build_args()
    data_frame = tech_indicators.read_df_from_file(user_args['name'])
    user_args["indicators"] = [s.strip() for s in user_args["indicators"].split(",")]
    indicator_df, buy_sell_hold_df = tech_indicators.get_indicators(data_frame, options=user_args["indicators"], length=user_args['length'])
    normalized_indicators_df = tech_indicators.normalize_indicators(indicator_df)

    X_train, X_test, Y_train, Y_test, = train_test_split(normalized_indicators_df, buy_sell_hold_df, shuffle=False, test_size=0.20)
    y_pred, rmse = train_and_predict(X_train, X_test, Y_train, Y_test)
    import pdb
    pdb.set_trace()
    generate_plots(y_pred, Y_test, rmse, user_args['name'], length_of_moving_averages=user_args['length'])
