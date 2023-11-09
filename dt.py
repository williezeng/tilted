from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import math
from models import BaseModel
from sklearn.utils import class_weight
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

LEAF_SAMPLES = list(range(1, 500, 10))
MAX_DEPTH = list(range(1, 100, 1))

DT_PARAMS = {'min_samples_leaf': 2,
             'min_samples_split': 2,
             'ccp_alpha': 0.0001,
             'max_depth': None,
             'max_features': 'sqrt',
             'criterion': 'entropy'}

PARAMETER_SPACE = {
                   "min_samples_leaf": hp.choice('min_samples_leaf', LEAF_SAMPLES),
                   "max_depth": hp.choice('max_depth', MAX_DEPTH),
                   }

PARAM_TO_LIST_MAP = {"min_samples_leaf": LEAF_SAMPLES,
                     "max_depth": MAX_DEPTH}

class DecisionTree(BaseModel):
    def __init__(self, options, data_frame):
        super().__init__(options, data_frame)
        self.model = DecisionTreeClassifier
        self.model_name = 'decision_tree'
        n_samples = len(self.ytrain)
        # self.weights = [i / n_samples for i in range(1, n_samples + 1)]

        if options.get('optimize_params'):
            self.params = DT_PARAMS.update(self.find_best_parameters(PARAMETER_SPACE))
        else:
            self.params = DT_PARAMS


    def find_best_parameters(self, parameter_space):
        trials = Trials()
        best_parameters = fmin(fn=self.optimize_params_score,
                               space=parameter_space,
                               algo=tpe.suggest,  # the logic which chooses next parameter to try
                               max_evals=1000,
                               trials=trials
                               )

        for k, v in best_parameters.items():
            if k in PARAM_TO_LIST_MAP:
                best_parameters[k] = PARAM_TO_LIST_MAP[k][v]
        print(best_parameters)
        return best_parameters

    def generate_plots(self):
        feature_names = self.xtrain.columns
        export_graphviz(self.model, out_file='tree.dot', class_names=['sell', 'hold', 'buy'], feature_names=feature_names, filled=True)
