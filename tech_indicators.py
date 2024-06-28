import numpy as np
from datetime import datetime

import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
TECHNICAL_INDICATORS = ['sma', 'ema', 'bb', 'high', 'low', 'volume', 'close', 'cmf', 'obv', 'rsi']
# 'bb_signal' is faulty
#
BUY = 1
SELL = -1
HOLD = 0


# TODO:  correlation, relative strength index (RSI), the difference between the open price of yesterday and today, difference close price of yesterday


def moving_average(data_frame, length):
    dataframe_copy = data_frame.copy()
    ema = dataframe_copy.ta.ema(length=length).dropna()
    sma = dataframe_copy.ta.sma(length=length).dropna()
    return ema, sma


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
    standard_deviation = data_frame.ta.stdev(length=length).dropna()
    bbstd = 2
    deviations = bbstd * standard_deviation
    lower_bb = moving_average - deviations
    upper_bb = moving_average + deviations
    return lower_bb, upper_bb


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

    list_of_dfs = []
    ema, sma = moving_average(df[['Close']], length)
    obv_vol = get_obv_vol(df)
    cmf_vol = get_cmf_vol(df, length)
    lower_bb_sma, upper_bb_sma = bbands_calculation(df[['Close']], sma, length)
    lower_bb_ema, upper_bb_ema = bbands_calculation(df[['Close']], ema, length)
    # averages are calculated given n previous days of information, drop the NAs
    df_close = df[['Close']].dropna()
    bb = pd.DataFrame({'lower_bb_sma': lower_bb_sma, 'upper_bb_sma': upper_bb_sma, 'lower_bb_ema': lower_bb_ema,
                       'upper_bb_ema': upper_bb_ema})

    y_label_df = create_ylabels(df_close[['Close']].astype(float), y_test_lookahead)

    bb_signal = bbands_classification(df_close, lower_bb_ema, upper_bb_ema)
    OPTION_MAP = {'sma': pd.DataFrame({'SMA_{}'.format(length): sma}),
                  'ema': pd.DataFrame({'EMA_{}'.format(length): ema}),
                  'bb': bb,
                  'high': df[['High']],
                  'low': df[['Low']],
                  'volume': df[['Volume']],
                  'close': df_close[['Close']],
                  'bb_signal': bb_signal,
                  'cmf': cmf_vol,
                  'obv': obv_vol,
                  'rsi': get_rsi(df, length),
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
        if ratio > 0.06:  # positive ratio that's higher than trade impact + commission
            trainY.append(BUY)
        elif ratio < -0.06:
            trainY.append(SELL)  # sell
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