from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import tech_indicators
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
import joblib
import os
import math
from abc import ABC, abstractmethod
from analyzer import check_buy_sell_signals

best = math.inf

BEST_RUN_DIR = os.path.join(os.path.curdir, 'best_run')
SAVED_MODEL = 'finalized_{name}.sav'
SAVED_MODEL_PATH = os.path.join(BEST_RUN_DIR, SAVED_MODEL)

class BaseModel(object):
    def __init__(self, options, data_frame):
        self.length_of_moving_averages = options.get('length')
        self.indicators = options.get('indicators')
        self.data_name = options.get('file_name')
        self.y_test_lookahead_days = options.get('lookahead_days')
        self.random_int_seed = options.get('random_seed')
        self.ypred = None
        self.model = None
        self.model_name = None
        self.rmse = None
        self.test_score = -1
        self.train_score = -1
        self.xtrain, self.xtest, self.ytrain, self.ytest, self.live_df = self.setup_data(data_frame)

    def optimize_parameters(self, options):
        if options.get('optimize_params'):
            self.params = self.find_best_parameters(self.create_parameter_space())
        else:
            self.params = None


    @abstractmethod
    def create_parameter_space(self):
        pass

    def accuracy_model(self, params):
        model_instance = self.model(**params)
        model_instance.fit(self.xtrain, self.ytrain['bs_signal'])
        y_pred = model_instance.predict(self.xtest)
        return mean_squared_error(self.ytest['bs_signal'], y_pred, squared=False)

    def f(self, params):
        global best
        acc = self.accuracy_model(params)
        if acc < best:
            best = acc
        return {'loss': acc, 'status': STATUS_OK}

    # Finding out which set of hyperparameters give highest accuracy
    @abstractmethod
    def find_best_parameters(self, parameter_space):
        pass

    def setup_data(self, data_frame):
        indicator_df, buy_sell_hold_df = tech_indicators.get_indicators(data_frame, self.indicators, self.length_of_moving_averages, self.y_test_lookahead_days)
        refined_indicators_df, refined_bs_df = tech_indicators.index_len_resolver(indicator_df, buy_sell_hold_df)
        normalized_indicators_df = tech_indicators.normalize_indicators(refined_indicators_df)
        # the buy_sell_hold_df will always END earlier than the indicator_df because of the lookahead days
        # the indicator_df will always START later than the buy_sell_hold_df because of the average_day length
        # bsh       =      [ , , , ]
        # indicator =         [ , , , , ]
        live_days = list(set(indicator_df.index) - set(buy_sell_hold_df.index))
        live_days.sort()
        xtrain, xtest, ytrain, ytest = train_test_split(normalized_indicators_df, refined_bs_df, shuffle=True, test_size=0.20, random_state=self.random_int_seed)
        return xtrain, xtest, ytrain, ytest, indicator_df.loc[live_days]


    def train(self):
        if self.params:
            self.model = self.model(**self.params)
        else:
            self.model = self.model()

        self.model.fit(self.xtrain, self.ytrain['bs_signal'])

    def simulation_predict(self):
        self.ypred = pd.DataFrame(self.model.predict(self.xtest), index=self.xtest.index, columns=['bs_signal']).sort_index()  # predicted
        self.ytest = self.ytest.sort_index()
        self.rmse = mean_squared_error(self.ytest, self.ypred, squared=False)
        self.train_score = self.model.score(self.xtrain, self.ytrain)
        self.test_score = accuracy_score(self.ypred, self.ytest)

    def live_predict(self):
        predicted_live_data = pd.DataFrame(self.model.predict(self.live_df), index=self.live_df.index, columns=['bs_signal']).sort_index()
        return predicted_live_data

    def train_and_predict(self):
        self.train()
        self.simulation_predict()

    def generate_plots(self):
        df2 = pd.DataFrame(data=self.ypred, index=self.ytest.index, columns=['predicted']).astype('float')
        df3 = self.ytest[['bs_signal']].astype('float').rename(columns={'Close': 'actual'})
        ax = df2.plot()
        df3.plot(ax=ax, title='bs_signal pred values vs real values', fontsize=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('{} bsh values'.format(self.data_name))
        plt.text(0.5, 0.5, 'rmse: ' + str(self.rmse), ha='center', va='center', fontsize='small')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("{}{}_{}_plot.png".format(self.data_name, self.length_of_moving_averages, self.model_name), dpi=500)

    def save_model(self):
        joblib.dump(self.model, SAVED_MODEL_PATH.format(name=self.model_name))

    def load_model(self):
        loaded_model = None
        if os.path.exists(SAVED_MODEL_PATH.format(name=self.model_name)):
            loaded_model = joblib.load(SAVED_MODEL_PATH.format(name=self.model_name))
        return loaded_model

    def save_best_model(self):
        #TODO: logic is flawed because the fucking dates are always shufflede lmao you are always comparing with different test results

        loaded_model = self.load_model()
        if loaded_model:
            loaded_model_y_pred = pd.DataFrame(loaded_model.predict(self.xtest), index=self.ytest.index, columns=['bs_signal']).sort_index()
            loaded_model_rmse = mean_squared_error(self.ytest, loaded_model_y_pred, squared=False)
            test_score = accuracy_score(loaded_model_y_pred, self.ytest)
            if test_score < self.test_score and loaded_model_rmse > self.rmse:
                print('---')
                print('Saving current model. It is better than this:')
                print('test acc ', test_score)
                check_buy_sell_signals(loaded_model_y_pred, self.ytest)
                self.save_model()
        else:
            self.save_model()

