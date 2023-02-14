import os
import numpy as np
from utils import constants
from datetime import datetime
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

BUY = 1
SELL = -1
HOLD = 0


# TODO:  correlation, relative strength index (RSI), the difference between the open price of yesterday and today, difference close price of yesterday


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


def bbands_classification(close_prices_data_frame, lower_bb, upper_bb):
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0
    closed_prices = close_prices_data_frame['Close']
    close_prices_data_frame = close_prices_data_frame.iloc[1:]  # base buy/sell off previous prices -1 len
    for i in range(1, len(closed_prices)):
        # BUY if dips lower than lower BB
        if closed_prices[i - 1] > lower_bb[i - 1] and closed_prices[i] < lower_bb[i]:
            if signal != 1:  # don't buy until sell
                buy_price.append(closed_prices[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        # SELL if rises above higher BB
        elif closed_prices[i - 1] < upper_bb[i - 1] and closed_prices[i] > upper_bb[i]:
            if signal != -1:  # don't sell until buy
                buy_price.append(np.nan)
                sell_price.append(closed_prices[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)

    close_prices_data_frame['bb_signal'] = bb_signal
    return close_prices_data_frame


def read_df_from_file(name):
    df = pd.read_csv(name, index_col=[0], header=[0], skipinitialspace=True)
    return df


def index_len_resolver(df1, df2):
    df1_start = datetime.strptime(df1.index[0], '%Y-%m-%d')
    df2_start = datetime.strptime(df2.index[0], '%Y-%m-%d')
    diff = (df2_start - df1_start).days
    if diff > 0:
        df1 = df1[diff:]
    elif diff < 0:
        df2 = df2[diff:]

    df1_end = datetime.strptime(df1.index[-1], '%Y-%m-%d')
    df2_end = datetime.strptime(df2.index[-1], '%Y-%m-%d')
    diff = (df2_end - df1_end).days

    if diff > 0:
        df2 = df2[:-diff]
    elif diff < 0:
        df1 = df1[:diff]
    return df1, df2


def get_indicators(df, options=None, length=10):
    df_close = df[['Close']]
    df_close['High'] = df[['High']]
    df_close['Volume'] = df[['Volume']]
    list_of_dfs = []
    ema, sma = moving_average(df_close, length=length)
    lower_bb_sma, upper_bb_sma = bbands_calculation(df_close, sma, length=length)
    lower_bb_ema, upper_bb_ema = bbands_calculation(df_close, ema, length=length)
    # averages are calculated given n previous days of information, drop the NAs
    df_close = df_close.dropna()
    bb = pd.DataFrame({'lower_bb_sma': lower_bb_sma, 'upper_bb_sma': upper_bb_sma, 'lower_bb_ema': lower_bb_ema,
                       'upper_bb_ema': upper_bb_ema})
    compiled_df = bbands_classification(df_close[['Close']][length - 1:].astype(float), bb['lower_bb_ema'],
                                        bb['upper_bb_ema'])
    y_label_df = create_ylabels(df_close[['Close']].astype(float))

    df_close, y_label_df = index_len_resolver(df_close, y_label_df)

    OPTION_MAP = {'sma': df_close['SMA_{}'.format(length)],
                  'ema': df_close['EMA_{}'.format(length)],
                  'bb': bb,
                  'high': df_close['High'],
                  'volume': df_close['Volume'],
                  'close': df_close['Close']
                  }
    for option in options:
        if option in OPTION_MAP:
            list_of_dfs.append(OPTION_MAP[option])
    X = pd.concat(list_of_dfs, axis=1)
    X = X.dropna()

    return index_len_resolver(X, y_label_df)
    # return X, y_label_df


def normalize_indicators(dfs):
    normalized_df = []
    for dataf in dfs:
        normalized_df.append(dfs[dataf].astype(float) / dfs[dataf].astype(float).iloc[0])
    return pd.concat(normalized_df, axis=1)


def create_ylabels(df, lookahead_days=5):
    # Returns buy/sell/hold signals
    # Shortens the data because our logic is based on the lookahead/future price
    trainY = []
    closed_price_series = df['Close']
    for i in range(closed_price_series.shape[0] - lookahead_days):
        ratio = (closed_price_series[i + lookahead_days] - closed_price_series[i]) / closed_price_series[i]
        if ratio > (0.02):  # positive ratio that's higher than trade impact + commission
            trainY.append(BUY)
        elif ratio < (-0.02):
            trainY.append(SELL)  # sell
        else:
            trainY.append(HOLD)
    df = df[:-lookahead_days]
    df['bs_signal'] = trainY
    return df[['bs_signal']]


def add_share_quantity(bs_df, amount_of_shares):
    # TODO: Write tests
    # we want to buy whatever we can with our starting value
    # we do this by calculating the most shares we can buy at the first buy signal
    # SELL is always selling everything
    # BUY is always buying as much as possible??????????????
    holdings = 0
    amount_to_switch_to_buy_or_sell = 2 * amount_of_shares
    entire_book_order = pd.DataFrame(index=bs_df.index, columns=['share_amount'])
    for index in range(len(bs_df)):
        if bs_df[index] == BUY and holdings < amount_of_shares:
            if holdings == 0:
                entire_book_order['share_amount'][index] = amount_of_shares
                holdings += amount_of_shares
            else:  # needed to switch from sell to buy
                entire_book_order['share_amount'][index] = amount_to_switch_to_buy_or_sell
                holdings += amount_to_switch_to_buy_or_sell
        # sell
        elif bs_df[index] == SELL and holdings > -amount_of_shares:
            if holdings == 0:
                entire_book_order['share_amount'][index] = amount_of_shares
                holdings = holdings - amount_of_shares
            else:  # needed to switch from buy to sell
                entire_book_order['share_amount'][index] = amount_to_switch_to_buy_or_sell
                holdings = holdings - amount_to_switch_to_buy_or_sell
    entire_book_order['bs_signal'] = bs_df
    checker = [(entire_book_order['share_amount'][x], entire_book_order.index[x]) for x in range(len(entire_book_order))
               if not np.isnan(entire_book_order['share_amount'][x])]
    print(checker)
    return entire_book_order
