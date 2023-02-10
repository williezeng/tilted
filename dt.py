from sklearn.tree import DecisionTreeClassifier
import pandas as pd

import matplotlib
matplotlib.use("Agg")

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

        print('with best params :', best_parameters)
        return best_parameters

# def train_and_predict(xtrain, xtest, ytrain, ytest, bestparams=None):
#     if isinstance(bestparams, dict):
#         model = DecisionTreeRegressor(**bestparams)
#     else:
#         model = DecisionTreeRegressor()
#     model.fit(xtrain, ytrain)
#     y_pred = model.predict(xtest)  # predicted
#     # Printout relevant metrics
#     rmse = mean_squared_error(ytest, y_pred, squared=False)
#     print("Mean Absolute Error: $", mean_absolute_error(ytest, y_pred))
#     print("Root Mean Square Error: $", rmse)
#
#     print("Coefficient of Determination:", r2_score(ytest, y_pred))
#     regression_confidence = model.score(xtest, ytest)
#     print("dt regression confidence: ", regression_confidence)
#     return y_pred, rmse
#
# def generate_plots(y_pred, Y_test, rmse, name, length_of_moving_averages=10):
#     df2 = pd.DataFrame(data=y_pred, index=Y_test.index, columns=['predicted']).astype('float')
#     df3 = Y_test[['Close']].astype('float').rename(columns={'Close':'actual'})
#     ax = df2.plot()
#     df3.plot(ax=ax, title='pred values vs real values', fontsize=10)
#
#     ax.set_xlabel('Date')
#     ax.set_ylabel('{} Price'.format(name))
#     plt.text(0.5, 0.5, 'rmse: '+str(rmse), ha='center', va='center', fontsize='small')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("{}{}_decision_tree_plot.png".format(name, length_of_moving_averages), dpi=500)
#
# def build_args():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--lowest_knn_neighbor', help='we need to separate search and knn', type=int)
#     parser.add_argument('--length', help='the length for moving averages', type=int, default=10)
#     parser.add_argument('--name', help='the name of the file', type=str, required=True)
#     parser.add_argument('--hyper_search', help='search for hyperparameters', type=bool, default=False)
#
#     return parser.parse_args()
#
# if __name__ == "__main__":
#     args = build_args()
#     data_frame = tech_indicators.read_df_from_file(args.name)
#     indicator_dfs, df_close = tech_indicators.get_indicators(data_frame, length=args.length)
#     normalized_indicators = tech_indicators.normalize_indicators(indicator_dfs)
#     # In other words, X_train and X_test is tech indicators
#     # Y_train and Y_test is close price
#     # axis=1 means horizontally concat
#     X = pd.concat(indicator_dfs, axis=1)
#     Y = df_close['Close'][args.length-1:]
#     X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, shuffle=False, test_size=0.20)
#     if args.hyper_search:
#         parameter_space = create_parameter_space()
#         best_parameters = find_best_parameters(parameter_space)
#     best_parameters = None
#     import pdb
#     pdb.set_trace()
#     y_pred, rmse = train_and_predict(X_train, X_test, Y_train, Y_test, best_parameters)
#
#
#     generate_plots(y_pred, Y_test, rmse, args.name, length_of_moving_averages=args.length)
