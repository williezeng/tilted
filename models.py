from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import tech_indicators
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal

import math
from abc import ABC, abstractmethod

best = math.inf


class BaseModel(object):
    def __init__(self, options, data_frame):
        self.length_of_moving_averages = options.get('length')
        self.indicators = options.get('indicators')
        self.data_name = options.get('file_name')

        self.ypred = None
        self.model = None
        self.model_name = None
        self.rmse = None
        self.xtrain, self.xtest, self.ytrain, self.ytest = self.setup_data(data_frame)

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
        indicator_df, buy_sell_hold_df = tech_indicators.get_indicators(data_frame, options=self.indicators,
                                                                        length=self.length_of_moving_averages)
        normalized_indicators_df = tech_indicators.normalize_indicators(indicator_df)
        return train_test_split(normalized_indicators_df, buy_sell_hold_df, shuffle=True, test_size=0.20)

    def train_and_predict(self):
        if self.params:
            model = self.model(**self.params)
        else:
            model = self.model()
        model.fit(self.xtrain, self.ytrain)
        self.ypred = pd.DataFrame(model.predict(self.xtest), index=self.ytest.index, columns=['bs_signal']).sort_index()  # predicted
        self.ytest = self.ytest.sort_index()
        self.rmse = mean_squared_error(self.ytest, self.ypred, squared=False)
        train_score = model.score(self.xtrain, self.ytrain)
        test_score = accuracy_score(self.ypred, self.ytest)
        print("Mean Absolute Error: $", self.rmse)
        print('test acc ', test_score)
        print('train acc ', train_score)

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
