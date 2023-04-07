from dt import DT_PARAMS

import matplotlib
matplotlib.use('TkAgg')

import math
from models import BaseModel

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
import pandas as pd

MAX_DEPTH = list(range(1, 100, 1))
RF_PARAMS = {'n_estimators': 200}
RF_PARAMS.update(DT_PARAMS)


class RandomForest(BaseModel):
    def __init__(self, options, data_frame):
        super().__init__(options, data_frame)
        self.model_name = 'random_forest'
        self.handle_params(options.get('optimize_params'))
        self.normalized_indicators_df, self.refined_bs_df, self.live_df = self.setup_data()
        self.xtrain, self.xtest, self.ytrain, self.ytest, self.ypred, self.train_score, self.test_score = self.train_and_predict()
        # self.ytest = self.ytest[-50:]
        # self.ypred = self.ypred[-50:]


    def live_predict(self):
        self.train(self.normalized_indicators_df, self.refined_bs_df)
        predicted_live_data = pd.DataFrame(self.model.predict(self.live_df), index=self.live_df.index, columns=['bs_signal'])
        return predicted_live_data

    def handle_params(self, optimize_params):
        if optimize_params:
            exit('NOT IMPLEMENTED')
            self.params = self.find_best_parameters(self.create_parameter_space())
        else:
            self.params = RF_PARAMS

    def create_parameter_space(self):
        parameter_space = {'max_depth': hp.choice('max_depth', MAX_DEPTH),

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
            if k == 'max_depth':
                best_parameters[k] = MAX_DEPTH[v]
        return best_parameters

