from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use('TkAgg')

import math
from models import BaseModel

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
LEAF_SAMPLES = list(range(1, 500, 10))
MAX_DEPTH = list(range(1, 100, 1))

DT_PARAMS = {'min_samples_leaf': 1,
             'max_depth': None}

PARAMETER_SPACE = {
                   "min_samples_leaf": hp.choice('min_samples_leaf', LEAF_SAMPLES),
                   "max_depth": hp.choice('max_depth', MAX_DEPTH)
                   }

PARAM_TO_LIST_MAP = {"min_samples_leaf": LEAF_SAMPLES,
                     "max_depth": MAX_DEPTH}

class DecisionTree(BaseModel):
    def __init__(self, options, data_frame):
        super().__init__(options, data_frame)
        self.model = DecisionTreeClassifier
        self.model_name = 'decision_tree'
        if options.get('optimize_params'):
            self.params = self.find_best_parameters(PARAMETER_SPACE)
        else:
            self.params = DT_PARAMS


    def find_best_parameters(self, parameter_space):
        trials = Trials()
        best_parameters = fmin(fn=self.f,
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

