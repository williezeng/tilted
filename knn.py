import os
from utils import external_ticks, constants
from sklearn.model_selection import train_test_split
# Machine learning libraries
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
import pandas as pd
import pandas_ta
from talib import BBANDS
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn import preprocessing
import tech_indicators
import argparse



# Instantiate KNN learning model(k=15)
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
# Defining the hyper parameter space as a dictionary

n_neighbor_list = list(range(5, 100))
leaf_size_list = list(range(1, 100))

def create_parameter_space():
    parameter_space = {'n_neighbors': hp.choice('n_neighbors', n_neighbor_list),
                       'weights': hp.choice('weights', ['distance']),
                       'algorithm': hp.choice('algorithm', ['brute', 'ball_tree', 'kd_tree']),
                       'leaf_size': hp.choice('leaf_size', leaf_size_list),
                       'p': hp.choice('p', [1, 2]),
                       'metric': hp.choice('metric', ['minkowski', 'chebyshev']),
                       }
    return parameter_space


best = math.inf


def f(params):
    global best
    acc = accuracy_model(params)
    if acc < best:
        best = acc
    return {'loss': acc, 'status': STATUS_OK}

def accuracy_model(params):
    knn = KNeighborsRegressor(**params)
    knn.fit(X_train, Y_train['Close'])
    y_pred = knn.predict(X_test)
    return mean_squared_error(Y_test['Close'], y_pred, squared=False)

# Finding out which set of hyperparameters give highest accuracy
def find_best_parameters(parameter_space):
    trials = Trials()
    best_parameters = fmin(fn=f,
                       space=parameter_space,
                       algo=tpe.suggest,  # the logic which chooses next parameter to try
                       max_evals=100,
                       trials=trials
                       )
    algo_map = {0: 'brute',
                1: 'ball_tree',
                2: 'kd_tree'}
    weight_map = {0: 'distance'}
    metric_map = {0: 'minkowski',
                  1: 'chebyshev'}
    for k, v in best_parameters.items():
        if k == 'algorithm':
            best_parameters[k] = algo_map[v]
        elif k == 'weights':
            best_parameters[k] = weight_map[v]
        elif k == 'metric':
            best_parameters[k] = metric_map[v]
        elif k == 'p':
            best_parameters[k] = v+1
        elif k == 'leaf_size':
            best_parameters[k] = leaf_size_list[v]
        elif k == 'n_neighbors':
            best_parameters[k] = n_neighbor_list[v]
    print('best: ', best, 'with best params :', best_parameters)
    return best_parameters


def train_and_predict(xtrain, xtest, ytrain, ytest, parameters):
    knn = KNeighborsRegressor(**parameters)
    knn.fit(xtrain, ytrain['Close'])
    y_pred = knn.predict(xtest)
    rmse = mean_squared_error(ytest['Close'], y_pred, squared=False)
    print("knn rmse: ", rmse)
    return y_pred, rmse
#
def generate_plots(y_predicted, Y_test, rmse, name, length_of_moving_averages=10):
    df2 = pd.DataFrame(data=y_predicted, index=Y_test.index, columns=['predicted']).astype('float')
    df_temp = pd.concat((pd.DataFrame(data=y_predicted, index=Y_test.index), Y_test['Close']), keys=['predicted', 'close'], axis=1)
    # print(df_temp.head)
    df3 = Y_test[['Close']].astype('float').rename(columns={'Close':'actual'})
    ax = df2.plot()
    df3.plot(ax=ax, title='pred values vs real values', fontsize=10)

    ax.set_xlabel('Date')
    ax.set_ylabel('{} Price'.format(name))
    plt.text(0.5, 0.5, 'rmse: '+str(rmse), ha='center', va='center', fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_name = "{}_knn_plot.png".format(length_of_moving_averages)
    plt.savefig(plot_name, dpi=500)
    print(plot_name)

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
    parser.add_argument('--name', help='the name of the file', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = build_args()
    name = os.path.join(constants.YAHOO_DATA_DIR, args.name)
    data_frame = tech_indicators.read_df_from_file(name)
    indicator_dfs, df_close = tech_indicators.get_indicators(data_frame, length=args.length)
    normalized_indicators = tech_indicators.normalize_indicators(indicator_dfs)
    # In other words, X_train and X_test is tech indicators
    # Y_train and Y_test is close price
    # axis=1 means horizontally concat
    X = pd.concat(normalized_indicators, axis=1)

    Y = df_close[['Close']][args.length-1:]
    Y = Y.astype(float)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.20)
    parameter_space = create_parameter_space()
    best_parameters = find_best_parameters(parameter_space)
    y_pred, rmse = train_and_predict(X_train, X_test, Y_train, Y_test, best_parameters)
    ticker_name = 'Ethereum'
    generate_plots(y_pred, Y_test, rmse, ticker_name, length_of_moving_averages=args.length)
