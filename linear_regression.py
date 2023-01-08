import os
from utils import external_ticks, constants
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
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
lower_bb_sma = pd.DataFrame(lower_bb_sma, columns=['lower_bb_sma'])
upper_bb_sma = pd.DataFrame(upper_bb_sma, columns=['upper_bb_sma'])

lower_bb_ema = pd.DataFrame(lower_bb_ema, columns=['lower_bb_ema'])
upper_bb_ema = pd.DataFrame(upper_bb_ema, columns=['upper_bb_ema'])
list_of_dfs = [df_close['SMA_{}'.format(length)][length-1:],
               df_close['EMA_{}'.format(length)][length-1:],
               lower_bb_sma,
               upper_bb_sma,
               lower_bb_ema,
               upper_bb_ema,
               df[['High']].iloc[2:][length-1:],
               df['Low'].iloc[2:][length-1:]]
normalized_df = []
for dataf in list_of_dfs:
    normalized_df.append(dataf.astype(float) / dataf.astype(float).iloc[0])

# generate_plot(normalized_df)


# In other words, X_train and X_test is tech indicators
# Y_train and Y_test is close price
# axis=1 means horizontally concat
X = pd.concat(normalized_df, axis=1)

Y = df_close[['Close']][length-1:]
X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, shuffle=False, test_size=0.20)

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)      # predicted
# Printout relevant metrics

print("Model Coefficients:", model.coef_)
print("Mean Absolute Error: $", mean_absolute_error(Y_test, y_pred))
print("Root Mean Square Error: $", mean_squared_error(Y_test, y_pred, squared=False))


print("Coefficient of Determination:", r2_score(Y_test, y_pred))
regression_confidence = model.score(X_test, Y_test)
print("linear regression confidence: ", regression_confidence)

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
plt.savefig("{}_linear_regression_plot.png".format(length), dpi=500)