from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib
import pandas as pd
from models import BaseModel
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
from sklearn.tree import export_graphviz
import math
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
        self.model_name = 'decision_tree'
        self.params = DT_PARAMS
        self.normalized_indicators_df, self.refined_bs_df, self.live_df = self.setup_data()
        self.xtrain, self.ytrain, self.xtest, self.ytest, self.ypred, self.train_score, self.test_score = self.train_and_predict()
        if options.get('optimize_params'):
            self.handle_params()
            self.xtrain, self.ytrain, self.xtest, self.ytest, self.ypred, self.train_score, self.test_score = self.train_and_predict()

    def live_predict(self):
        self.train(self.normalized_indicators_df, self.refined_bs_df)
        predicted_live_data = pd.DataFrame(self.model.predict(self.live_df), index=self.live_df.index,
                                           columns=['bs_signal'])
        return predicted_live_data

    def handle_params(self):
        self.params.update(self.find_best_parameters(self.create_parameter_space()))

    def create_parameter_space(self):
        parameter_space = {'max_depth': hp.choice('max_depth', MAX_DEPTH),

                           }
        return parameter_space

    best = math.inf


    def find_best_parameters(self, parameter_space):
        trials = Trials()
        best_parameters = fmin(fn=self.optimize_params_score,
                               space=parameter_space,
                               algo=tpe.suggest,  # the logic which chooses next parameter to try
                               max_evals=100,
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
