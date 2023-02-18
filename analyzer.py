import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime
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

    percentage_buys = (correct_buys / total_buys) * 100
    percentage_sells = (correct_sells / total_sells) * 100
    print('---------------------')
    print('there are supposed to be {} buys. there are supposed to be {} sells'.format(total_buys, total_sells))
    print('you made {} correct buys. you made {} correct sells'.format(correct_buys, correct_sells))
    print('correct buy percentage {} and correct sell percentage {}'.format(percentage_buys, percentage_sells))


def compute_portvals(order_book, close_data, start_val, commission=9.95, impact=0.005):
    order_book_copy = order_book.copy()
    order_book_copy = order_book_copy.dropna()
    filled_orders = pd.DataFrame(index=order_book_copy.index, columns=['holding', 'cash_earned', 'cumulative_gains', 'cumulative_percent'])
    shares_holder = 0
    gains_holder = 0
    start_date = order_book_copy.index[0]
    end_date = order_book_copy.index[-1]

    for index in order_book_copy.index:
        if order_book_copy.loc[index, 'bs_signal'] == BUY:
            shares_holder += order_book_copy.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_data.loc[index, 'Close'] * abs(order_book_copy.loc[index, 'share_amount']) * (
                        1.000 + impact)) + commission)
            filled_orders['cash_earned'][index] = -cost
            gains_holder -= cost
            filled_orders['cumulative_gains'][index] = gains_holder
            filled_orders['cumulative_percent'][start_date] = (gains_holder - filled_orders['cumulative_gains'][
                start_date]) / abs(filled_orders['cumulative_gains'][start_date]) * 100

        elif order_book_copy.loc[index, 'bs_signal'] == SELL:
            shares_holder -= order_book_copy.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_data.loc[index, 'Close'] * abs(order_book_copy.loc[index, 'share_amount']) * (
                        1.000 - impact)) - commission)
            filled_orders['cash_earned'][index] = cost
            gains_holder += cost
            filled_orders['cumulative_gains'][index] = gains_holder
            filled_orders['cumulative_percent'][end_date] = (gains_holder - filled_orders['cumulative_gains'][start_date])/abs(filled_orders['cumulative_gains'][start_date]) * 100

    order_book_copy['cash_earned'] = filled_orders['cash_earned']
    order_book_copy['holding'] = filled_orders['holding']
    order_book_copy['cumulative_gains'] = filled_orders['cumulative_gains']
    order_book_copy['cumulative_percent'] = filled_orders['cumulative_percent']

    order_book_copy['close'] = close_data['Close']
    order_book_copy = order_book_copy.dropna()
    print('model net profits {} from {} to {}'.format(order_book_copy.sum()['cash_earned'], order_book_copy.index[0], order_book_copy.index[-1]))
    return order_book_copy


def compute_simple_baseline(close_df, share_amount, start_val, commission=9.95, impact=0.005):
    """
    Computes the value of holding the share amnt on the first day
    INPUT: BUY/SELL orderbook DO NOT use LONG/SHORTS
    """
    close_df_copy = close_df.copy()
    close_df_copy = close_df_copy.dropna()
    start_date = close_df_copy.index[0]
    end_date = close_df_copy.index[-1]
    filled_orders = pd.DataFrame(index=close_df_copy.index, columns=['holding', 'cash_earned', 'cumulative_gains', 'cumulative_percent'])
    shares_holder = 0
    gains_holder = 0

    # Execute buy
    shares_holder += share_amount
    filled_orders['holding'][start_date] = shares_holder
    cost = float((close_df_copy['Close'][0] * share_amount * (1.000 + impact)) + commission)
    filled_orders['cash_earned'][start_date] = -cost
    gains_holder -= cost
    filled_orders['cumulative_gains'][start_date] = gains_holder
    filled_orders['cumulative_percent'][start_date] = (gains_holder - filled_orders['cumulative_gains'][start_date])/abs(filled_orders['cumulative_gains'][start_date]) * 100
    # Execute sell
    shares_holder -= share_amount
    filled_orders['holding'][end_date] = shares_holder
    cost = float((close_df_copy['Close'][-1] * share_amount * (1.000 + impact)) + commission)
    filled_orders['cash_earned'][end_date] = cost
    gains_holder += cost
    filled_orders['cumulative_gains'][end_date] = gains_holder
    filled_orders['cumulative_percent'][end_date] = (gains_holder - filled_orders['cumulative_gains'][start_date])/abs(filled_orders['cumulative_gains'][start_date]) * 100

    summed = filled_orders.sum()
    filled_orders = filled_orders.dropna()
    print('----')
    print('If you bought {} shares on {}, you would have {} on {}'.format(filled_orders['holding'][0], start_date,
                                                                          summed['cash_earned'], end_date))
    percent_increase = (close_df_copy['Close'][-1] - close_df_copy['Close'][0]) / abs(close_df_copy['Close'][0]) * 100
    print('The share increased by {} %'.format(percent_increase))
    return filled_orders


def compute_spy(spy_df, share_amount, start_val, commission=9.95, impact=0.005):
    spy_df_copy = spy_df.copy()
    spy_df_copy = spy_df_copy.dropna()
    start_date = spy_df_copy.index[0]
    end_date = spy_df_copy.index[-1]
    filled_orders = pd.DataFrame(index=spy_df_copy.index, columns=['holding', 'cash_earned', 'cumulative_gains', 'cumulative_percent'])
    shares_holder = 0
    gains_holder = 0
    # Execute buy
    shares_holder += share_amount
    filled_orders['holding'][start_date] = shares_holder
    cost = float((spy_df['Close'][0] * share_amount * (1.000 + impact)) + commission)
    filled_orders['cash_earned'][start_date] = -cost
    gains_holder -= cost
    filled_orders['cumulative_gains'][start_date] = gains_holder
    filled_orders['cumulative_percent'][start_date] = (gains_holder - filled_orders['cumulative_gains'][start_date])/abs(filled_orders['cumulative_gains'][start_date]) * 100

    # Execute sell
    shares_holder -= share_amount
    filled_orders['holding'][end_date] = shares_holder
    cost = float((spy_df['Close'][-1] * share_amount * (1.000 + impact)) + commission)
    filled_orders['cash_earned'][end_date] = cost
    gains_holder += cost
    filled_orders['cumulative_gains'][end_date] = gains_holder
    filled_orders['cumulative_percent'][end_date] = (gains_holder - filled_orders['cumulative_gains'][start_date])/abs(filled_orders['cumulative_gains'][start_date]) * 100

    summed = filled_orders.sum()
    filled_orders = filled_orders.dropna()
    print('----')
    print('If you bought {} shares of SPY on {}, you would have {} on {}'.format(filled_orders['holding'][0], start_date, summed['cash_earned'], end_date))
    percent_increase = (spy_df['Close'][-1] - spy_df['Close'][0]) / abs(spy_df['Close'][0]) * 100
    print('SPY increased by {} %'.format(percent_increase))
    return filled_orders


def compute_yearly_gains(order_book):
    yearly_gains_dict = {}
    order_book_start = datetime.strptime(order_book.index[0], '%Y-%m-%d')
    order_book_end = datetime.strptime(order_book.index[-1], '%Y-%m-%d')
    x = 1
    date_holder = '{}-{}-{}'.format(order_book_start.year, order_book_start.month, order_book_start.day)
    next_years_date = '{}-{}-{}'.format(order_book_start.year + x, order_book_start.month, order_book_start.day)
    while int(next_years_date.split('-')[0]) <= int(order_book_end.year):
        yearly_gain = ((order_book.loc[date_holder:next_years_date].sum()['cash_earned'] - order_book.loc[:date_holder]['cash_earned'][-1])/abs(order_book.loc[:date_holder]['cash_earned'][-1]) * 100)
        print('From {} to {} the yearly gain is {}'.format(date_holder, next_years_date, yearly_gain))
        date_holder = next_years_date
        yearly_gains_dict[x] = yearly_gain
        x += 1
        next_years_date = '{}-{}-{}'.format(order_book_start.year + x, order_book_start.month, order_book_start.day)

    order_book_end_str = '{}-{}-{}'.format(order_book_end.year, order_book_end.month, order_book_end.day)
    remaining_gains = order_book.loc[date_holder:order_book_end_str].sum()['cash_earned']
    if remaining_gains > 0.0:
        yearly_gain = ((order_book.loc[date_holder:order_book_end_str].sum()['cash_earned'] -
                        order_book.loc[:date_holder]['cash_earned'][-1]) / abs(order_book.loc[:date_holder]['cash_earned'][-1]) * 100)
        yearly_gains_dict[x] = yearly_gain

    total_percent_gain = ((order_book['cash_earned'].sum() - order_book['cash_earned'][0]) / abs(
        order_book['cash_earned'][0])) * 100
    print('THE TOTAL % gain : {}'.format(total_percent_gain))
    return yearly_gains_dict

def compare_strategies(buy_sell_order_book, long_short_order_book, target_close_df, spy_close_df, args):
    print('----')
    print('generating longs and shorts ')
    long_short_portfolio_values = compute_portvals(long_short_order_book, target_close_df, args['starting_value'])
    graph_order_book(long_short_portfolio_values, target_close_df, args['model_name'], args['file_name'],
                     args["indicators"], args['length'])
    compute_yearly_gains(long_short_portfolio_values)

    print('generating buy and sell')
    buy_sell_portfolio_values = compute_portvals(buy_sell_order_book, target_close_df, args['starting_value'])
    target_hold_portfolio = compute_simple_baseline(target_close_df, int(args['share_amount']), args['starting_value'])
    spy_hold_portfolio = spy_portfolio_values = compute_spy(spy_close_df, int(args['share_amount']), args['starting_value'], commission=0, impact=0)
    import pdb
    pdb.set_trace()
    graph_spy(spy_portfolio_values, spy_close_df)
    # compute_yearly_gains(buy_sell_portfolio_values)
    # compute_yearly_gains(target_hold_portfolio)
    # compute_yearly_gains(spy_hold_portfolio)


def graph_spy(spy_order_book, spy_close):
    plt.plot(list(spy_close.index), spy_close['Close'], label='price')
    plt.plot(spy_order_book.index[0], spy_close['Close'][0], '--go', label='buy')
    plt.plot(spy_order_book.index[1], spy_close['Close'][-1], '--ro', label='sell')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
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
    sell_markers_date = [order_book.index[x] for x in range(len(order_book)) if order_book['bs_signal'][x] == SELL]
    sell_markers_price = [order_book['close'][x] for x in range(len(order_book)) if order_book['bs_signal'][x] == SELL]

    buy_markers_date = [order_book.index[x] for x in range(len(order_book)) if order_book['bs_signal'][x] == BUY]
    buy_markers_price = [order_book['close'][x] for x in range(len(order_book)) if order_book['bs_signal'][x] == BUY]

    plt.plot(list(close_data.index), close_data['Close'], label='price')
    plt.plot(sell_markers_date, sell_markers_price, '--ro', label='sell')
    plt.plot(buy_markers_date, buy_markers_price, '--go', label='buy')

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=90, fontweight='light', fontsize='x-small')
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
