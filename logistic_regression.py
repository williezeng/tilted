import os
from utils import external_ticks, constants
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
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
from models import BaseModel

tol = [0.00001, 0.0001, 0.001, 0.01, 0.1]
# , 'l1', 'elasticnet'
# penalty = [None, 'l2']
c_values = [1, 2, 3, 4, 5, 6, 7]
solver = ['newton-cholesky']


#
#     ‘newton-cg’ - [‘l2’, None]
#     ‘newton-cholesky’ - [‘l2’, None]
#     ‘lbfgs’ - [‘l2’, None]
#     ‘liblinear’ - [‘l1’, ‘l2’]
#     ‘sag’ - [‘l2’, None]
#     ‘saga’ - [‘elasticnet’, ‘l1’, ‘l2’, None]
class TiltedLogisticRegression(BaseModel):
    def __init__(self, options, data_frame):
        super().__init__(options, data_frame)
        self.model = LogisticRegression
        self.model_name = 'logistic regression'
        self.optimize_parameters(options)


    def create_parameter_space(self):
        # , 'l1', 'elasticnet'
        parameter_space = {'solver': hp.choice('solver', solver),
                           'tol': hp.choice('tol', tol),
                           'C': hp.choice('C', c_values),
                           }
        return parameter_space

    def find_best_parameters(self, parameter_space):
        trials = Trials()
        best_parameters = fmin(fn=self.f,
                               space=parameter_space,
                               algo=tpe.suggest,  # the logic which chooses next parameter to try
                               max_evals=100,
                               trials=trials
                               )
        for k, v in best_parameters.items():
            if k == 'tol':
                best_parameters[k] = tol[v]
            elif k == 'C':
                best_parameters[k] = c_values[v]
            elif k == 'solver':
                best_parameters[k] = solver[v]

        # best_params['penalty'] = 'l2'
        print('with best params :', best_parameters)
        return best_parameters
