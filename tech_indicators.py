import numpy as np
from datetime import datetime

import pandas
import pandas as pd
import matplotlib
from utils import constants
from utils.constants import BUY, SELL, HOLD

matplotlib.use('TkAgg')
TECHNICAL_INDICATORS = ['SMA_10', 'EMA_10', 'bb_upper', 'bb_lower', 'bb_width', 'close']
# 'bb_signal,  'obv', 'rsi'' is faulty
#


# TODO:  correlation, relative strength index (RSI), the difference between the open price of yesterday and today, difference close price of yesterday

# TODO: CREATE STRATEGY: SMA + BOELI + MACD:
# # Create a new DataFrame 'data' with the 'Close' column from 'data_frame'
# data = pd.DataFrame()
# data['Close'] = data_frame['Close']
#
# # Calculate technical indicators
# data['EMA50'] = data['Close'].ta.ema(length=50).dropna()
# data['EMA200'] = data['Close'].ta.ema(length=200).dropna()
# data['MACD'] = data['Close'].ta.macd().dropna()['MACD_12_26_9']
# data['RSI'] = data['Close'].ta.rsi(length=14).dropna()
#
# # Drop NaN values that result from indicator calculation
# data = data.dropna()
#
# # Signal generation based on combined indicators
# data['Signal'] = 0
#
# # Buy signal: EMA50 > EMA200, MACD > 0, RSI < 70
# data.loc[(data['EMA50'] > data['EMA200']) & (data['MACD'] > 0) & (data['RSI'] < 70), 'Signal'] = 1
#
# # Sell signal: EMA50 < EMA200, MACD < 0, RSI > 30
# data.loc[(data['EMA50'] < data['EMA200']) & (data['MACD'] < 0) & (data['RSI'] > 30), 'Signal'] = -1
#
# # Generate Position column to reflect holding periods
# data['Position'] = data['Signal']
#
# # Print the DataFrame with signals
# print(data.tail())
#
# # Plotting
# plt.figure(figsize=(14, 7))
# plt.plot(data['Close'], label='Close Price', alpha=0.35)
# plt.plot(data['EMA50'], label='50-Day EMA', alpha=0.75)
# plt.plot(data['EMA200'], label='200-Day EMA', alpha=0.75)
#
# # Plot buy signals
# plt.scatter(data.index, data['Close'], c=np.where(data['Position'] == 1, 'green', np.nan), label='Buy Signal', marker='^', alpha=1.0)
#
# # Plot sell signals
# plt.scatter(data.index, data['Close'], c=np.where(data['Position'] == -1, 'orange', np.nan), label='Sell Signal', marker='v', alpha=1.0)
#
# plt.title('Entry and Exit Signals using EMA, MACD, and RSI')
# plt.legend()
# plt.show()

def moving_sma(data_frame):
    return data_frame


def moving_ema(data_frame):

    #
    # data['Signal'] = 0
    #
    # # Generate signals for crossovers
    # data['Signal'] = np.where(
    #     (data['Close'] > data['Long_EMA']) & (data['Close'].shift(1) <= data['Long_EMA'].shift(1)), 1,
    #     np.where((data['Close'] < data['Long_EMA']) & (data['Close'].shift(1) >= data['Long_EMA'].shift(1)), -1,
    #              0)
    # )
    #
    # plt = create_stock_graph(data['Close'], data['Signal'])
    # plt.plot(list(data.index), data['Long_EMA'], label='Long_EMA')
    # plt.legend()
    # plt.show()
    return data_frame['Long_EMA']


def get_obv_vol(data_frame):
    dataframe_copy = data_frame.copy()
    obv = dataframe_copy.ta.obv().dropna()
    return obv


def get_cmf_vol(data_frame, length):
    dataframe_copy = data_frame.copy()
    cmf = dataframe_copy.ta.cmf(length=length).dropna()
    return cmf


def get_rsi(data_frame, length):
    # It oscillates between 0 and 100,
    # readings above 70 indicating overbought conditions
    # readings below 30 indicating oversold conditions
    dataframe_copy = data_frame.copy()
    rsi = dataframe_copy.ta.rsi(length=length).dropna()
    return rsi


def bbands_calculation(data_frame, moving_average, length):
    # imported pandasta bbands calculations are broken, lingering na's in their sma implementation
    # input should be some sort of moving average, df
    # standard_deviation = data_frame.ta.stdev(length=length).dropna()
    # bbstd = 2
    # deviations = bbstd * standard_deviation
    # lower_bb = moving_average - deviations
    # upper_bb = moving_average + deviations
    dataframe_copy = data_frame.copy()
    return



def bbands_classification(close_prices_data_frame, lower_bb, upper_bb):
    close_prices_data_frame = close_prices_data_frame.iloc[1:]  # base buy/sell off previous prices -1 len
    upper_bb, close_prices_data_frame = index_len_resolver(upper_bb, close_prices_data_frame)
    upper_bb, lower_bb = index_len_resolver(upper_bb, lower_bb)
    close_prices_df_copy = close_prices_data_frame.copy()
    bb_signal = pd.DataFrame(index=close_prices_df_copy.index, columns=['bb_signal'])
    for i in range(1, len(close_prices_df_copy['Close'])):
        # BUY if dips lower than lower BB
        if close_prices_df_copy['Close'][i - 1] >= lower_bb[i - 1] and close_prices_df_copy['Close'][i] <= lower_bb[i]:
            bb_signal['bb_signal'][i] = 1
        # BUY if rises past lower BB
        elif close_prices_df_copy['Close'][i - 1] <= lower_bb[i - 1] and close_prices_df_copy['Close'][i] >= lower_bb[i]:
            bb_signal['bb_signal'][i] = 1

        # SELL if rises above higher BB
        elif close_prices_df_copy['Close'][i - 1] <= upper_bb[i - 1] and close_prices_df_copy['Close'][i] >= upper_bb[i]:
            bb_signal['bb_signal'][i] = -1
        # SELL if dips below higher BB
        elif close_prices_df_copy['Close'][i - 1] >= upper_bb[i - 1] and close_prices_df_copy['Close'][i] <= upper_bb[i]:
            bb_signal['bb_signal'][i] = -1
        else:
            bb_signal['bb_signal'][i] = 0
    return bb_signal[1:]



def index_len_resolver(df1, df2):
    df1_start = datetime.strptime(df1.index[0], '%Y-%m-%d')
    df2_start = datetime.strptime(df2.index[0], '%Y-%m-%d')
    diff = (df2_start - df1_start).days
    if diff > 0:
        df1 = df1[df2.index[0]:]
    elif diff < 0:
        df2 = df2[df1.index[0]:]

    df1_end = datetime.strptime(df1.index[-1], '%Y-%m-%d')
    df2_end = datetime.strptime(df2.index[-1], '%Y-%m-%d')
    diff = (df2_end - df1_end).days

    if diff > 0:
        df2 = df2[:df1.index[-1]]
    elif diff < 0:
        df1 = df1[:df2.index[-1]]
    return df1, df2


def get_indicators(df, options, length, y_test_lookahead):
    df_copy = df.copy()
    df_copy['SMA_10'] = (df_copy['Close'] - df_copy.ta.ema(length=constants.LONG_TERM_PERIOD).dropna())/df_copy['Close']
    df_copy['EMA_10'] = (df_copy['Close'] - df_copy.ta.ema(length=constants.LONG_TERM_PERIOD).dropna())/df_copy['Close']
    bb_results = df_copy.ta.bbands(length=constants.LONG_TERM_PERIOD).dropna()
    #     lower=BBL_{length}_{std},  mid = BBM_{length}_{std}, upper = BBU_{length}_{std}
    #     bandwidth = BBB_{length}_{std}, percent = BBP_{length}_{std}
    df_copy = df_copy.join(bb_results, how='inner')
    df_copy['bb_upper'] = (df_copy['BBU_10_2.0'] - df_copy['Close']) / df_copy['Close']
    df_copy['bb_lower'] = (df_copy['BBL_10_2.0'] - df_copy['Close']) / df_copy['Close']
    df_copy['bb_width'] = (df_copy['BBU_10_2.0'] - df_copy['BBL_10_2.0']) / df_copy['Close']
    list_of_dfs = []
    # averages are calculated given n previous days of information, drop the NAs
    y_label_df = create_ylabels(df[['Close']].astype(float), y_test_lookahead)
    OPTION_MAP = {'SMA_10': df_copy['SMA_10'],
                  'EMA_10': df_copy['EMA_10'],
                  'bb_upper': df_copy['bb_upper'],
                  'bb_lower': df_copy['bb_lower'],
                  'bb_width': df_copy['bb_width'],
                  # 'high': df[['High']],
                  # 'low': df[['Low']],
                  'close': df_copy['Close'],
                  'volume': df_copy['Volume'],
                  # 'bb_signal': bb_signal,
                  # 'cmf': cmf_vol,
                  # 'obv': obv_vol,
                  # 'rsi': get_rsi(df, length),
                  }


    for option in options:
        if option in OPTION_MAP:
            list_of_dfs.append(OPTION_MAP[option])

    x = pd.concat(list_of_dfs, axis=1)
    x = x.dropna()
    return x, y_label_df


def normalize_indicators(dfs):
    normalized_df = []
    for dataf in dfs:
        if set(dfs[dataf].values) == {0, 1, -1}:
            normalized_df.append(dfs[dataf].astype(float))
        else:
            normalized_df.append(dfs[dataf].astype(float) / dfs[dataf].astype(float).iloc[0])
    return pd.concat(normalized_df, axis=1)


def create_ylabels(df, lookahead_days):
    """
    Creates the Y labels (Buy/Sell/Hold) for training
    :param df:
    :param lookahead_days:
    :return: dataframe with Y labels
    """
    # Returns buy/sell/hold signals
    # Shortens the data because our logic is based on the lookahead/future price
    # The lookahead days depends on every stock ticker, slower stocks would need a longer day
    trainY = []
    df_copy = df.copy()
    closed_price_series = df_copy['Close']
    for i in range(closed_price_series.shape[0] - lookahead_days):
        ratio = (closed_price_series[i + lookahead_days] - closed_price_series[i]) / closed_price_series[i]
        # a larger buy threshold will result to fewer trades.
        if ratio > constants.BUY_THRESHOLD:
            trainY.append(BUY)
        elif ratio < constants.SELL_THRESHOLD:
            trainY.append(SELL)
        else:
            trainY.append(HOLD)
    df_copy = df_copy[:-lookahead_days]
    df_copy['bs_signal'] = trainY
    return df_copy[['bs_signal']]


def add_long_short_shares(bs_df, amount_of_shares):
    """
    long 200 to fill our shorts and hold 100
    short 200 to sell our 100 and hold 100 shares that we don't have
    the shares that we borrow will be filled by the long 200
    """
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
    entire_book_order = entire_book_order.dropna()
    return entire_book_order


def add_buy_sell_shares(bs_df, close_price, starting_value, offset=0.008, impact=0.005):
    holdings = 0
    entire_book_order = pd.DataFrame(index=bs_df.index, columns=['share_amount'])
    gains_holder = starting_value
    for index in bs_df.index:
        if bs_df[index] == BUY and holdings == 0:
            # 0.90 for risk management
            number_of_buyable_shares = (gains_holder * 0.95) / close_price.loc[index][0]
            gains_holder -= (close_price.loc[index][0] * number_of_buyable_shares)
            entire_book_order['share_amount'][index] = number_of_buyable_shares
            holdings += number_of_buyable_shares
        elif bs_df[index] == SELL and holdings > 0:
            entire_book_order['share_amount'][index] = holdings
            gains_holder = (close_price.loc[index][0] * holdings)
            holdings -= holdings
    entire_book_order['bs_signal'] = bs_df
    entire_book_order = entire_book_order.dropna()
    return entire_book_order


def setup_data(df_from_ticker, indicators, length, lookahead_days):
    indicator_df, buy_sell_hold_df = get_indicators(df_from_ticker, indicators, length, lookahead_days)
    refined_indicators_df, refined_bs_df = index_len_resolver(indicator_df, buy_sell_hold_df)
    normalized_indicators_df = normalize_indicators(refined_indicators_df)
    # the buy_sell_hold_df will always END earlier than the indicator_df because of the lookahead days
    # the indicator_df will always START later than the buy_sell_hold_df because of the average_day length
    # bsh       =      [ , , , ]
    # indicator =         [ , , , , ]
    future_prediction_days = list(set(indicator_df.index) - set(buy_sell_hold_df.index))
    future_prediction_days.sort()
    return normalized_indicators_df, refined_bs_df, indicator_df.loc[future_prediction_days]
