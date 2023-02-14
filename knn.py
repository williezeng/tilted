import os
from utils import external_ticks, constants
from sklearn.model_selection import train_test_split
# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
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
from models import BaseModel


# Instantiate KNN learning model(k=15)
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
# Defining the hyper parameter space as a dictionary

n_neighbor_list = list(range(5, 100))
leaf_size_list = list(range(1, 100))
algo_map = {0: 'brute',
            1: 'ball_tree',
            2: 'kd_tree'}
weight_map = {0: 'distance'}
metric_map = {0: 'minkowski',
              1: 'chebyshev'}


class KNN(BaseModel):
    def __init__(self, options, data_frame):
        super().__init__(options, data_frame)
        self.model = KNeighborsClassifier
        self.model_name = 'knn'
        self.optimize_parameters(options)

    def create_parameter_space(self):
        parameter_space = {'n_neighbors': hp.choice('n_neighbors', n_neighbor_list),
                           'weights': hp.choice('weights', ['distance']),
                           'algorithm': hp.choice('algorithm', algo_map.values()),
                           'leaf_size': hp.choice('leaf_size', leaf_size_list),
                           'p': hp.choice('p', [1, 2]),
                           'metric': hp.choice('metric', metric_map.values()),
                           }
        return parameter_space


    # Finding out which set of hyperparameters give highest accuracy
    def find_best_parameters(self, parameter_space):
        trials = Trials()
        best_parameters = fmin(fn=self.f,
                           space=parameter_space,
                           algo=tpe.suggest,  # the logic which chooses next parameter to try
                           max_evals=100,
                           trials=trials
                           )

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
        print('best params :', best_parameters)
        return best_parameters
