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
        self.model_name = 'knn'
        self.params = {'n_neighbors': 1}

        self.normalized_indicators_df, self.refined_bs_df, self.live_df = self.setup_data()
        self.xtrain, self.ytrain, self.xtest, self.ytest, self.ypred, self.train_score, self.test_score = self.train_and_predict()
        if options.get('optimize_params'):
            self.handle_params()
    def create_parameter_space(self):
        parameter_space = {'n_neighbors': hp.choice('n_neighbors', n_neighbor_list),
                           'weights': hp.choice('weights', ['distance']),
                           'algorithm': hp.choice('algorithm', algo_map.values()),
                           'leaf_size': hp.choice('leaf_size', leaf_size_list),
                           'p': hp.choice('p', [1, 2]),
                           'metric': hp.choice('metric', metric_map.values()),
                           }
        return parameter_space

    def live_predict(self):
        self.train(self.normalized_indicators_df, self.refined_bs_df)
        predicted_live_data = pd.DataFrame(self.model.predict(self.live_df), index=self.live_df.index,
                                           columns=['bs_signal'])
        return predicted_live_data

    def handle_params(self):
        self.params = self.find_best_parameters(self.create_parameter_space())

    # Finding out which set of hyperparameters give highest accuracy
    def find_best_parameters(self, parameter_space):
        trials = Trials()
        best_parameters = fmin(fn=self.optimize_params_score,
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
                best_parameters[k] = v + 1
            elif k == 'leaf_size':
                best_parameters[k] = leaf_size_list[v]
            elif k == 'n_neighbors':
                best_parameters[k] = n_neighbor_list[v]
        return best_parameters
