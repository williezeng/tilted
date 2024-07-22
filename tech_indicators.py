import numpy as np
from datetime import datetime

import pandas as pd
import pandas_ta as ta
import matplotlib
from utils import constants
from utils.constants import BUY, SELL, HOLD
import talib

matplotlib.use('TkAgg')

# TODO:  correlation, relative strength index (RSI), the difference between the open price of yesterday and today, difference close price of yesterday

# TODO: CREATE STRATEGY: SMA + BOELI + MACD:


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


def index_len_resolver(df1, df2):
    # Ensure the indices are datetime
    if not isinstance(df1.index, pd.DatetimeIndex):
        df1.index = pd.to_datetime(df1.index, format='%Y-%m-%d')
    if not isinstance(df2.index, pd.DatetimeIndex):
        df2.index = pd.to_datetime(df2.index, format='%Y-%m-%d')

    # Find the overlapping date range
    start_date = max(df1.index[0], df2.index[0])
    end_date = min(df1.index[-1], df2.index[-1])

    # Slice the dataframes to the overlapping date range
    df1 = df1[start_date:end_date]
    df2 = df2[start_date:end_date]
    return df1, df2


def calculate_vwap(df, window=10):
    """
    Calculate the VWAP for a given window.

    Parameters:
    df (pd.DataFrame): Dataframe containing 'close', 'volume', and optionally 'high' and 'low' columns.
    window (int): Number of days for the VWAP calculation.

    Returns:
    pd.Series: VWAP values.
    """
    # Ensure the dataframe contains the required columns
    if not {'Close', 'Volume'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'close' and 'volume' columns.")

    # Calculate the typical price
    if {'High', 'Low'}.issubset(df.columns):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    else:
        typical_price = df['Close']

    # Calculate the cumulative typical price * volume
    cumulative_tpv = (typical_price * df['Volume']).cumsum()

    # Calculate the cumulative volume
    cumulative_volume = df['Volume'].cumsum()

    # Calculate VWAP
    vwap = cumulative_tpv / cumulative_volume

    return vwap


def calculate_amo(df,lookback=constants.LOOK_BACK_PERIOD):
    # Calculate momentum
    momentum = df['Close'] - df['Close'].shift(lookback)
    # Calculate the absolute value of momentum
    abs_momentum = np.abs(momentum)
    # Calculate the average of the absolute momentum for the long period
    long_avg = abs_momentum.rolling(window=constants.LONG_TERM_PERIOD).mean()
    # Calculate the Adaptive Momentum Oscillator
    amo = momentum / long_avg
    return amo


def get_indicators(df, options, length, y_test_lookahead):
    df_copy = df.copy()
    df_copy['SMA_10'] = (df_copy['Close'] - df_copy.ta.sma(length=constants.LONG_TERM_PERIOD).dropna())/df_copy['Close']
    df_copy['EMA_10'] = (df_copy['Close'] - df_copy.ta.ema(length=constants.LONG_TERM_PERIOD).dropna())/df_copy['Close']
    bb_results = df_copy.ta.bbands(length=constants.LONG_TERM_PERIOD).dropna()
    #     lower=BBL_{length}_{std},  mid = BBM_{length}_{std}, upper = BBU_{length}_{std}
    #     bandwidth = BBB_{length}_{std}, percent = BBP_{length}_{std}
    df_copy = df_copy.join(bb_results, how='inner')
    df_copy['bb_upper'] = (df_copy['BBU_10_2.0'] - df_copy['Close']) / df_copy['Close']
    df_copy['bb_lower'] = (df_copy['BBL_10_2.0'] - df_copy['Close']) / df_copy['Close']
    df_copy['bb_width'] = (df_copy['BBU_10_2.0'] - df_copy['BBL_10_2.0']) / df_copy['Close']
    df_copy['VWAP'] = calculate_vwap(df_copy)
    df_copy['VWMA_10'] = (df_copy['VWAP'] - df_copy.ta.vwma(length=constants.LONG_TERM_PERIOD).dropna())/df_copy['VWAP']
    df_copy['RSI'] = df_copy.ta.rsi(length=constants.LONG_TERM_PERIOD)
    df_copy['AMO'] = calculate_amo(df_copy)
    df_copy = df_copy.dropna()
    if df_copy['RSI'].isna().any():
        print('THIS ticker has NaN RSI values')
    list_of_dfs = []
    # averages are calculated given n previous days of information, drop the NAs
    y_label_df = create_ylabels(df[['Close']].astype(float))

    options_map = {'SMA_10': df_copy['SMA_10'],
                   'EMA_10': df_copy['EMA_10'],
                   'bb_upper': df_copy['bb_upper'],
                   'bb_lower': df_copy['bb_lower'],
                   'bb_width': df_copy['bb_width'],
                   'VWMA_10': df_copy['VWMA_10'],
                   'VWAP': df_copy['VWAP'],
                   'RSI': df_copy['RSI'],
                   'AMO': df_copy['AMO'],
                   'Close': df_copy['Close'],
                   }
                   # 'volume': df_copy['Volume'],
                   # 'bb_signal': bb_signal,
                  # 'cmf': cmf_vol,
                  # 'obv': obv_vol,
                  # 'rsi': get_rsi(df, length),
    for option in options:
        if option in options_map:
            list_of_dfs.append(options_map[option])

    concat_dfs = pd.concat(list_of_dfs, axis=1)
    concat_dfs = concat_dfs.dropna()
    return concat_dfs, y_label_df


def normalize_indicators(name, dfs):
    normalized_df = []
    for dataf in dfs:
        df = dfs[dataf].astype(float)
        if set(dfs[dataf].values) == {0, 1, -1}:
            normalized_df.append(df)
        else:
            if df.iloc[0] == 0:
                print(f"Warning: The first value in the DataFrame {name} is zero, normalization skipped.")
                normalized_df.append(df)
                continue
            normalized_df.append(df / df.iloc[0])
    return pd.concat(normalized_df, axis=1)


def create_ylabels(df):
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
    for i in range(closed_price_series.shape[0] - constants.LOOK_AHEAD_DAYS_TO_GENERATE_BUY_SELL):
        ratio = (closed_price_series[i + constants.LOOK_AHEAD_DAYS_TO_GENERATE_BUY_SELL] - closed_price_series[i]) / closed_price_series[i]
        # a larger buy threshold will result to fewer trades.
        if ratio > constants.BUY_THRESHOLD:
            trainY.append(BUY)
        elif ratio < constants.SELL_THRESHOLD:
            trainY.append(SELL)
        else:
            trainY.append(HOLD)
    df_copy = df_copy[:-constants.LOOK_AHEAD_DAYS_TO_GENERATE_BUY_SELL]
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


def setup_data(index, df_from_ticker, indicators, length, lookahead_days):
    indicator_df, buy_sell_hold_df = get_indicators(df_from_ticker, indicators, length, lookahead_days)
    refined_indicators_df, refined_bs_df = index_len_resolver(indicator_df, buy_sell_hold_df)
    normalized_indicators_df = normalize_indicators(df_from_ticker.name, refined_indicators_df)
    # the buy_sell_hold_df will always END earlier than the indicator_df because of the lookahead days
    # the indicator_df will always START later than the buy_sell_hold_df because of the average_day length
    # bsh       =      [ , , , ]
    # indicator =         [ , , , , ]
    future_prediction_days = list(set(indicator_df.index) - set(buy_sell_hold_df.index))
    future_prediction_days.sort()
    return normalized_indicators_df, refined_bs_df, indicator_df.loc[future_prediction_days]
