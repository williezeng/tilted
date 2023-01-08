import os
from utils import external_ticks, constants
from sklearn.model_selection import train_test_split
# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pandas_ta
from talib import BBANDS
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#TODO:  correlation, relative strength index (RSI), the difference between the open price of yesterday and today, difference close price of yesterday


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

def read_df_from_file(name):
    name = os.path.join(constants.YAHOO_DATA_DIR, 'real_eth.csv')
    df = pd.read_csv(name, index_col=[0], header=[0], skipinitialspace=True)
    return df

def get_indicators(df, length=10):
    df_close = df[['Close']].iloc[2:]
    ema, sma = moving_average(df_close, length=length)
    lower_bb_sma, upper_bb_sma = bbands_calculation(df_close, sma, length=length)
    lower_bb_ema, upper_bb_ema = bbands_calculation(df_close, ema, length=length)
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
    return list_of_dfs, df_close

def normalize_indicators(list_of_dfs):
    normalized_df = []
    for dataf in list_of_dfs:
        normalized_df.append(dataf.astype(float) / dataf.astype(float).iloc[0])
    return normalized_df

