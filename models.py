from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import tech_indicators
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import os
import math
from abc import ABC, abstractmethod
from analyzer import check_buy_sell_signals
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree

best = math.inf

BEST_RUN_DIR = os.path.join(os.path.curdir, 'best_run')
SAVED_MODEL = 'finalized_{name}.sav'
SAVED_MODEL_PATH = os.path.join(BEST_RUN_DIR, SAVED_MODEL)
MODEL_NAME_TO_CLASSIFIER = {'random_forest': RandomForestClassifier,
                            'knn': KNeighborsClassifier,
                            'decision_tree': DecisionTreeClassifier}


class BaseModel(object):
    def __init__(self, options, data_frame):
        self.live_df = None
        self.refined_bs_df = None
        self.normalized_indicators_df = None
        self.length_of_moving_averages = options.get('length')
        self.data_frame = data_frame
        self.indicators = options.get('indicators')
        self.data_name = options.get('file_name')
        self.y_test_lookahead_days = options.get('lookahead_days')
        self.random_int_seed = options.get('random_seed')
        self.weights = None
        self.model = None
        self.model_name = None
        self.rmse = None
        self.test_score = -1
        self.train_score = -1
        self.params = None


    def optimize_parameters(self, options):
        if options.get('optimize_params'):
            self.params = self.find_best_parameters(self.create_parameter_space())
        else:
            self.params = None


    @abstractmethod
    def create_parameter_space(self):
        pass

    def accuracy_model(self, params):
        model_instance = MODEL_NAME_TO_CLASSIFIER[self.model_name](**self.params)
        model_instance.fit(self.xtrain, self.ytrain['bs_signal'])
        y_pred = model_instance.predict(self.xtest)
        return mean_squared_error(self.ytest['bs_signal'], y_pred, squared=False)

    def optimize_params_score(self, params):
        global best
        acc = self.accuracy_model(params)
        if acc < best:
            best = acc
        return {'loss': acc, 'status': STATUS_OK}

    # Finding out which set of hyperparameters give highest accuracy
    @abstractmethod
    def find_best_parameters(self, parameter_space):
        pass

    def setup_data(self):
        indicator_df, buy_sell_hold_df = tech_indicators.get_indicators(self.data_frame, self.indicators, self.length_of_moving_averages, self.y_test_lookahead_days)
        refined_indicators_df, refined_bs_df = tech_indicators.index_len_resolver(indicator_df, buy_sell_hold_df)
        normalized_indicators_df = tech_indicators.normalize_indicators(refined_indicators_df)
        # the buy_sell_hold_df will always END earlier than the indicator_df because of the lookahead days
        # the indicator_df will always START later than the buy_sell_hold_df because of the average_day length
        # bsh       =      [ , , , ]
        # indicator =         [ , , , , ]
        future_prediction_days = list(set(indicator_df.index) - set(buy_sell_hold_df.index))
        future_prediction_days.sort()
        return normalized_indicators_df, refined_bs_df, indicator_df.loc[future_prediction_days]

    def train(self, xtrain, ytrain):
        if self.params is not None:
            print(self.params)
            self.model = MODEL_NAME_TO_CLASSIFIER[self.model_name](**self.params)
        else:
            exit('NO PARAMS?')
        if self.weights:
            self.model.fit(xtrain, ytrain['bs_signal'], sample_weight=self.weights)
        else:
            self.model.fit(xtrain, ytrain['bs_signal'])

    def train_and_predict(self):
        # predictions = pd.DataFrame()
        # cumulative_training_indicator_data = pd.DataFrame()
        # cumulative_train_bs_data = pd.DataFrame()
        # cumulative_test_indicator_data = pd.DataFrame()
        # correct_test_output = pd.DataFrame()
        # k = 2      # chunks
        # # we want to keep a training to test ratio
        # # Calculate the chunk size
        # chunk_size = len(self.normalized_indicators_df) // k
        # chunk_indices = np.arange(len(self.normalized_indicators_df)) // chunk_size
        # Assuming you have a DataFrame or numpy arrays for your data and labels
        # X = your_data  # Features
        # y = your_labels  # Target variable
        # Split the data into training and testing sets with an 80:20 ratio

        X_train, X_test, y_train, y_test = train_test_split(self.normalized_indicators_df, self.refined_bs_df, test_size=0.2, shuffle=False)
        training_indices = list(range(0, len(X_train)))
        # np.random.seed(self.random_int_seed)
        # np.random.shuffle(training_indices)
        # shuffled_xtrain = X_train.iloc[training_indices]
        # shuffled_ytrain = y_train.iloc[training_indices]
        self.train(X_train, y_train)
        ypred = pd.DataFrame(self.model.predict(X_test), index=X_test.index, columns=['bs_signal'])

        # for fold in range(0, k):
        #     # Get the indices for the current fold, reserve the last lookahead indices for testing
        #     # df =  [[      Training          ][ Testing  ]] [[      Training          ][ Testing  ]]
        #     training_indices = np.where(chunk_indices == fold)[0][:-self.y_test_lookahead_days]
        #     test_indices = np.where(chunk_indices == fold)[0][-self.y_test_lookahead_days:]
        #     np.random.shuffle(training_indices)
        #     ytrain, ytest = self.refined_bs_df.iloc[training_indices], self.refined_bs_df.iloc[test_indices]
        #     cumulative_training_indicator_data = pd.concat([cumulative_training_indicator_data, xtrain])
        #     cumulative_train_bs_data = pd.concat([cumulative_train_bs_data, ytrain])
        #     #TODO: Do we want to repredict previous predictions? or just move forward? Currently we're just predicting forward
        #     ypred = pd.DataFrame(self.model.predict(xtest), index=xtest.index, columns=['bs_signal'])
        #     cumulative_test_indicator_data =  pd.concat([cumulative_test_indicator_data, xtest])
        #     correct_test_output = pd.concat([correct_test_output, ytest])
        #     predictions = pd.concat([predictions, ypred])

        return X_train, y_train, X_test, y_test, ypred, self.model.score(X_train, y_train), accuracy_score(y_test, ypred)

    def generate_plots(self):
        pass

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

