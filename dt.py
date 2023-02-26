from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use('TkAgg')

import math
from models import BaseModel

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
leaf_samples = list(range(5, 500, 10))

class DecisionTree(BaseModel):
    def __init__(self, options, data_frame):
        super().__init__(options, data_frame)
        self.model = DecisionTreeClassifier
        self.model_name = 'decision_tree'
        self.optimize_parameters(options)

    def create_parameter_space(self):
        parameter_space = {'min_samples_leaf': hp.choice('min_samples_leaf', leaf_samples),

                           }
        return parameter_space

    best = math.inf


    def find_best_parameters(self, parameter_space):
        trials = Trials()
        best_parameters = fmin(fn=self.f,
                           space=parameter_space,
                           algo=tpe.suggest,  # the logic which chooses next parameter to try
                           max_evals=500,
                           trials=trials
                           )

        for k, v in best_parameters.items():
            if k == 'min_samples_leaf':
                best_parameters[k] = leaf_samples[v]
        return best_parameters

