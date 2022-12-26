import os
from utils import external_ticks, constants
from sklearn.model_selection import train_test_split
# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pandas_ta
from talib import BBANDS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


name = os.path.join(constants.YAHOO_DATA_DIR, 'real_eth.csv')

def moving_average(data_frame, length=10):
    ema = data_frame.ta.ema(length=length, append=True).dropna()
    sma = data_frame.ta.sma(length=length, append=True).dropna()
    return ema, sma

def bbands_calculation(data_frame, moving_average, length=10):
    # imported pandasta bbands calculations are broken, lingering na's in their sma implementation
    # input should be some sort of moving average, df
    standard_deviation = data_frame.ta.stdev(length=length).dropna()
    bbstd = 2
    deviations = bbstd * standard_deviation
    lower_bb = moving_average - deviations
    upper_bb = moving_average + deviations
    return lower_bb, upper_bb

def generate_plot(dataframes):
    df_temp = pd.concat(dataframes, keys=['close', 'ema', 'sma', 'lowerbb_sma', 'upperbb_sma', 'lowerbb_ema', 'upperbb_ema'], axis=1)
    ax = df_temp.plot(title='Daily Portfolio Value and SPY', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Value')
    plt.savefig("plot.png", dpi=1000)

df = pd.read_csv(name, index_col=[0], header=[0], skipinitialspace=True)
length = 10
df_close = df[['Close']].iloc[2:]
list_of_dataframes = [df_close[length-1:], ]
ema, sma = moving_average(df_close, length=length)
lower_bb_sma, upper_bb_sma = bbands_calculation(df_close, sma, length=length)
lower_bb_ema, upper_bb_ema = bbands_calculation(df_close, ema, length=length)
list_of_dataframes.extend([ema, sma, lower_bb_sma, upper_bb_sma, lower_bb_ema, upper_bb_ema])

normalized_df = []
for dataf in list_of_dataframes:
    division = dataf.astype(float) / dataf.iloc[0].astype(float)

    normalized_df.append(division)

# generate_plot(normalized_df)


# In other words, X_train and X_test is tech indicators
# Y_train and Y_test is close price
# axis=1 means horizontally concat
X_train, X_test, Y_train, Y_test, = train_test_split(pd.concat((df_close['SMA_10'][length-1:], df_close['EMA_10'][length-1:]), axis=1), df_close[['Close']][length-1:], shuffle=False, test_size=0.2)

# Instantiate KNN learning model(k=15)
# knn = KNeighborsClassifier(n_neighbors=3)
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal
# Defining the hyper parameter space as a dictionary
parameter_space = {'n_neighbors': hp.quniform('n_neighbors', 1, 3, 5, 7, 9, 11),
                   'weights': hp.quniform('weights', ['uniform', 'distance']),
                   'algorithm': hp.choice('algorithm', ['brute']),
                   'p': hp.choice('p', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                   'metric': hp.choice('p', ['minkowski', 'chebyshev']),
                   }
'leaf_size': hp.choice('leaf_size', ['5', '10', '15', '20']),
'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),

knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)

# # Accuracy Score
# accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
# accuracy_test = accuracy_score(Y_test, y_pred)
#
# print ('Train_data Accuracy: %.2f' %accuracy_train)
# print ('Test_data Accuracy: %.2f' %accuracy_test)

print("Mean Absolute Error: $", mean_absolute_error(Y_test, y_pred))
print("Root Mean Square Error: $", mean_squared_error(Y_test, y_pred, squared=False))
print("Coefficient of Determination:", r2_score(Y_test, y_pred))
# regression_confidence = knn.score(X_test, Y_test)
# print("linear regression confidence: ", regression_confidence)

df2 = pd.DataFrame(data=y_pred, index=Y_test.index, columns=['predicted']).astype('float')
# df_temp = pd.concat((pd.DataFrame(data=y_pred, index=Y_test.index), Y_test['Close']), keys=['predicted', 'close'], axis=1)
# print(df_temp.head)
df3 = Y_test[['Close']].astype('float').rename(columns={'Close':'actual'})
ax = df2.plot()
df3.plot(ax=ax, title='pred values vs real values', fontsize=10)

name = 'Ethereum'
ax.set_xlabel('Date')
ax.set_ylabel('{} Price'.format(name))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("knn_plot.png", dpi=500)