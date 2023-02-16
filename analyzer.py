import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.dates as mdates

import matplotlib.pyplot as plt
from tech_indicators import BUY, SELL, HOLD

def check_buy_sell_signals(ypred, ytest):
    assert isinstance(ypred, pd.DataFrame)
    assert isinstance(ytest, pd.DataFrame)
    correct_buys = 0
    correct_sells = 0
    total_len = len(ypred)
    total_buys = len([x for x in ytest['bs_signal'] if x == 1])
    total_sells = len([x for x in ytest['bs_signal'] if x == -1])
    for x in range(total_len):
        if ypred['bs_signal'][x] == ytest['bs_signal'][x]:
            if ypred['bs_signal'][x] == 1:  # buy
                correct_buys += 1
            elif ypred['bs_signal'][x] == -1:  # sell
                correct_sells += 1

    percentage_buys = (correct_buys/total_buys) * 100
    percentage_sells = (correct_sells/total_sells) * 100
    print('---------------------')
    print('there are supposed to be {} buys. there are supposed to be {} sells'.format(total_buys, total_sells))
    print('you made {} correct buys. you made {} correct sells'.format(correct_buys, correct_sells))
    print('correct buy percentage {} and correct sell percentage {}'.format(percentage_buys, percentage_sells))

def compute_portvals(order_book, close_data, start_val, commission=9.95, impact=0.005):
    order_book_copy = order_book.copy()
    start_date = order_book_copy.index[0]
    end_date = order_book_copy.index[-1]
    # start_index = list(close_data.index).index(start_date)
    # end_index = list(close_data.index).index(end_date)
    # close_data = close_data[start_index:end_index]
    # book_order['close'] = close_data
    filled_orders = pd.DataFrame(index=order_book_copy.index, columns=['holding', 'total_liquid_cash'])
    filled_orders['total_liquid_cash'][0] = start_val
    shares_holder = 0
    for index in order_book_copy.index:
        if order_book_copy.loc[index, 'bs_signal'] == BUY and not np.isnan(order_book_copy.loc[index, 'share_amount']):
            shares_holder += order_book_copy.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_data.loc[index, 'Close'] * abs(order_book_copy.loc[index, 'share_amount']) * (1.000 + impact)) + commission)
            filled_orders['total_liquid_cash'][index] = -cost
        elif order_book_copy.loc[index, 'bs_signal'] == SELL and not np.isnan(order_book_copy.loc[index, 'share_amount']):
            shares_holder -= order_book_copy.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_data.loc[index, 'Close'] * abs(order_book_copy.loc[index, 'share_amount']) * (1.000 - impact)) - commission)
            filled_orders['total_liquid_cash'][index] = cost
    order_book_copy['total_liquid_cash'] = filled_orders['total_liquid_cash']
    order_book_copy['holding'] = filled_orders['holding']
    order_book_copy['close'] = close_data['Close']
    order_book_copy = order_book_copy.dropna()
    print('model net profits {} from {} to {}'.format(order_book_copy.sum()['total_liquid_cash'], start_date, end_date))
    return order_book_copy

def compute_simple_baseline(bs_orderbook, close_df, share_amount, start_val, commission=9.95, impact=0.005):
    """
    Computes the value of holding the share amnt on the first day
    INPUT: BUY/SELL orderbook DO NOT use LONG/SHORTS
    """
    order_book_copy = bs_orderbook.copy()
    order_book_copy = order_book_copy.dropna()

    start_date = bs_orderbook.index[0]
    end_date = bs_orderbook.index[-1]
    filled_orders = pd.DataFrame(index=order_book_copy.index, columns=['holding', 'total_liquid_cash'])
    filled_orders['total_liquid_cash'][0] = start_val
    shares_holder = 0
    # Execute buy
    shares_holder += share_amount
    filled_orders['holding'][start_date] = shares_holder
    cost = float((close_df.loc[start_date, 'Close'] * share_amount * (1.000 + impact)) + commission)
    filled_orders['total_liquid_cash'][start_date] = -cost
    # Execute sell
    shares_holder -= share_amount
    filled_orders['holding'][end_date] = shares_holder
    sell_cost = float((close_df.loc[end_date, 'Close'] * share_amount * (1.000 + impact)) + commission)
    filled_orders['total_liquid_cash'][end_date] = sell_cost

    filled_orders = filled_orders.dropna()
    summed = filled_orders.sum()
    print('If you bought {} shares on {}, you would have {} on {}'.format(filled_orders['holding'][0], start_date, summed['total_liquid_cash'], end_date))
    return filled_orders

def compute_spy(spy_df, share_amount, start_val, commission=9.95, impact=0.005):
    spy_df_copy = spy_df.copy()
    spy_df_copy = spy_df_copy.dropna()
    start_date = spy_df_copy.index[0]
    end_date = spy_df_copy.index[-1]
    filled_orders = pd.DataFrame(index=spy_df_copy.index, columns=['holding', 'total_liquid_cash'])
    filled_orders['total_liquid_cash'][0] = start_val
    shares_holder = 0
    # Execute buy
    shares_holder += share_amount
    filled_orders['holding'][start_date] = shares_holder
    cost = float((spy_df['Close'][0] * share_amount * (1.000 + impact)) + commission)
    filled_orders['total_liquid_cash'][start_date] = -cost
    # Execute sell
    shares_holder -= share_amount
    filled_orders['holding'][end_date] = shares_holder
    sell_cost = float((spy_df['Close'][-1] * share_amount * (1.000 + impact)) + commission)
    filled_orders['total_liquid_cash'][end_date] = sell_cost
    summed = filled_orders.sum()
    filled_orders = filled_orders.dropna()
    print('----')
    print('If you bought {} shares of SPY on {}, you would have {} on {}'.format(filled_orders['holding'][0], start_date, summed['total_liquid_cash'], end_date))
    percent_increase = (spy_df['Close'][-1] - spy_df['Close'][0])/spy_df['Close'][0] * 100
    print('SPY increased by {} %'.format(percent_increase))
    return filled_orders


def compare_strategies(buy_sell_order_book, long_short_order_book, target_close_df, spy_close_df, args):
    print('generating longs and shorts ')
    long_short_portfolio_values = compute_portvals(long_short_order_book, target_close_df, args['starting_value'])
    print('generating buy and sell')
    buy_sell_portfolio_values = compute_portvals(buy_sell_order_book, target_close_df, args['starting_value'])
    compute_simple_baseline(buy_sell_order_book, target_close_df, args['share_amount'], args['starting_value'])
    spy_portfolio_values = compute_spy(spy_close_df, args['share_amount'], args['starting_value'], commission=0, impact=0)
    graph_spy(spy_portfolio_values, spy_close_df)
    graph_order_book(long_short_portfolio_values, target_close_df, args['model_name'], args['file_name'], args["indicators"], args['length'])

def graph_spy(spy_order_book, spy_close):
    plt.plot(list(spy_close.index), spy_close['Close'], label='price')
    plt.plot(spy_order_book.index[0], spy_close['Close'][0], '--go', label='buy')
    plt.plot(spy_order_book.index[1], spy_close['Close'][-1], '--ro', label='sell')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=90, fontweight='light',  fontsize='x-small')
    plt.tight_layout()
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    plt.title('spy')
    figure = plt.gcf()
    figure.set_size_inches(20, 13)
    plt.tight_layout()
    plt.savefig('{}.png'.format('spy'), dpi=300)
    plt.figure().clear()


def graph_order_book(order_book, close_data, model_name, file_name, list_of_indicators, moving_average_length):
    # figure = plt.figure()
    # ax = close_data.plot(y='Close', x='Date')

    # start_date = order_book.index[0]
    # start_index = list(close_data.index).index(start_date)
    # test_close_data = close_data[start_index:]

    # diff = list(set(close_data.index) - set(order_book.index))
    # dropped_unrelclose_data.drop(diff)
    sell_markers_date = [order_book.index[x] for x in range(len(order_book)) if order_book['bs_signal'][x] == SELL]
    sell_markers_price = [order_book['close'][x] for x in range(len(order_book)) if order_book['bs_signal'][x] == SELL]

    buy_markers_date = [order_book.index[x] for x in range(len(order_book)) if order_book['bs_signal'][x] == BUY]
    buy_markers_price = [order_book['close'][x] for x in range(len(order_book)) if order_book['bs_signal'][x] == BUY]

    plt.plot(list(close_data.index), close_data['Close'], label='price')
    plt.plot(sell_markers_date, sell_markers_price, '--ro', label='sell')
    plt.plot(buy_markers_date, buy_markers_price, '--go', label='buy')

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=90, fontweight='light',  fontsize='x-small')
    plt.tight_layout()
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    plt.text(1, 0, '{}'.format(list_of_indicators), ha='right', va='bottom', fontsize='small')
    title = '{}_{}_{}'.format(model_name, file_name, moving_average_length)
    plt.title(title)
    figure = plt.gcf()
    figure.set_size_inches(20, 13)
    plt.tight_layout()
    plt.savefig('{}.png'.format(title), dpi=300)
    plt.figure().clear()