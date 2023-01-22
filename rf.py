import os
from utils import constants
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
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


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
min_samples_leaf = list(range(1, 20, 1)) # default = 1
min_samples_split = list(range(2, 10)) # default = 2
number_of_trees = list(range(50, 700, 10)) # default = 100
max_depth = list(range(1, 20, 1))
bootstrap = [True, False]


def create_parameter_space():
    parameter_space = {'min_samples_leaf': hp.choice('min_samples_leaf', min_samples_leaf),
                       'min_samples_split': hp.choice('min_samples_split', min_samples_split),
                       'n_estimators': hp.choice('n_estimators', number_of_trees),
                       'bootstrap': hp.choice('bootstrap', bootstrap),
                       'max_depth': hp.choice('max_depth', max_depth)}
    return parameter_space

best = math.inf

def accuracy_model(params):
    dtreg = RandomForestRegressor(**params)
    dtreg.fit(X_train, Y_train['Close'])
    y_pred = dtreg.predict(X_test)
    return mean_squared_error(Y_test['Close'], y_pred, squared=False)

def f(params):
    global best
    acc = accuracy_model(params)
    if acc < best:
        best = acc
    return {'loss': acc, 'status': STATUS_OK}

def find_best_parameters(parameter_space):
    trials = Trials()
    best_parameters = fmin(fn=f,
                       space=parameter_space,
                       algo=tpe.suggest,  # the logic which chooses next parameter to try
                       max_evals=700,
                       trials=trials
                       )

    for k, v in best_parameters.items():
        if k == 'min_samples_leaf':
            best_parameters[k] = min_samples_leaf[v]
        elif k == ' min_samples_split':
            best_parameters[k] = min_samples_split[v]
        elif k == 'n_estimators':
            best_parameters[k] = number_of_trees[v]
        elif k == 'bootstrap':
            best_parameters[k] = bootstrap[v]
        elif k == 'max_depth':
            best_parameters[k] = max_depth[v]
    print('best: ', best, 'with best params :', best_parameters)
    return best_parameters

def train_and_predict(xtrain, xtest, ytrain, ytest, bestparams=None):
    if isinstance(bestparams, dict):
        model = RandomForestRegressor(**bestparams)
    else:
        raise('wtf! no params')
    model.fit(xtrain, ytrain['Close'])
    y_pred = model.predict(xtest)  # predicted
    # Printout relevant metrics
    rmse = mean_squared_error(ytest, y_pred, squared=False)
    print("Mean Absolute Error: $", mean_absolute_error(ytest, y_pred))
    print("Root Mean Square Error: $", rmse)

    print("Coefficient of Determination:", r2_score(ytest, y_pred))
    regression_confidence = model.score(xtest, ytest)
    print("dt regression confidence: ", regression_confidence)
    return y_pred, rmse

def generate_plots(y_pred, Y_test, rmse, name, length_of_moving_averages=10):
    df2 = pd.DataFrame(data=y_pred, index=Y_test.index, columns=['predicted']).astype('float')
    df3 = Y_test[['Close']].astype('float').rename(columns={'Close':'actual'})
    ax = df2.plot()
    df3.plot(ax=ax, title='pred values vs real values', fontsize=10)

    ax.set_xlabel('Date')
    ax.set_ylabel('{} Price'.format(name))
    plt.text(0.5, 0.5, 'rmse: '+str(rmse), ha='center', va='center', fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("{}{}_random_forest_plot.png".format(name, length_of_moving_averages), dpi=500)

def build_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--lowest_knn_neighbor', help='we need to separate search and knn', type=int)
    parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
    parser.add_argument('--name', help='the name of the file', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = build_args()
    data_frame = tech_indicators.read_df_from_file(args.name)
    indicator_dfs, df_close = tech_indicators.get_indicators(data_frame, length=args.length)
    normalized_indicators = tech_indicators.normalize_indicators(indicator_dfs)
    # In other words, X_train and X_test is tech indicators
    # Y_train and Y_test is close price
    # axis=1 means horizontally concat
    X = pd.concat(normalized_indicators, axis=1)
    Y = df_close[['Close']][args.length-1:]
    X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, shuffle=False, test_size=0.20)
    parameter_space = create_parameter_space()
    best_parameters = find_best_parameters(parameter_space)
    y_pred, rmse = train_and_predict(X_train, X_test, Y_train, Y_test, best_parameters)

    generate_plots(y_pred, Y_test, rmse, args.name, length_of_moving_averages=args.length)
