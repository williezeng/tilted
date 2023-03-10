import os
import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
from utils import trading_logger
import matplotlib.pyplot as plt
from tech_indicators import BUY, SELL, HOLD, add_long_short_shares, add_buy_sell_shares


RECENT_RUN_DIR = os.path.join(os.path.curdir, 'recent_run')
logger = trading_logger.getlogger()
OUTPUT = []


def analyze_prediction_to_test(ypred, ytest, dataframe_from_tickers, args):
    check_buy_sell_signals(ypred, ytest)
    compute_best_case(ytest, dataframe_from_tickers, args['share_amount'], args['starting_value'], args['lookahead_days'], args['save_recent'])


def check_buy_sell_signals(ypred, ytest):
    assert isinstance(ypred, pd.DataFrame)
    assert isinstance(ytest, pd.DataFrame)
    correct_buys = 0
    correct_sells = 0
    total_len = len(ypred)
    total_buys = len([x for x in ytest['bs_signal'] if x == 1])
    total_predicted_buys = len([x for x in ypred['bs_signal'] if x == 1])

    total_sells = len([x for x in ytest['bs_signal'] if x == -1])
    total_predicted_sells = len([x for x in ypred['bs_signal'] if x == -1])

    for x in range(total_len):
        if ypred['bs_signal'][x] == ytest['bs_signal'][x]:
            if ypred['bs_signal'][x] == 1:  # buy
                correct_buys += 1
            elif ypred['bs_signal'][x] == -1:  # sell
                correct_sells += 1

    percentage_buys = (correct_buys / total_buys) * 100
    percentage_sells = (correct_sells / total_sells) * 100
    output = [
        'there are supposed to be {} trades. The model made {} trades.'.format(total_buys+total_sells, total_predicted_buys+total_predicted_sells),
        'there are supposed to be {} buys. there are supposed to be {} sells'.format(total_buys, total_sells),
        'you made {} correct buys. you made {} correct sells'.format(correct_buys, correct_sells),
        'you made {} incorrect buys. you made {} incorrect sells'.format(total_predicted_buys-correct_buys, total_predicted_sells-correct_sells),
        'correct buy percentage {} and correct sell percentage {}'.format(percentage_buys, percentage_sells)
        ]
    OUTPUT.extend(output)


def compute_best_case(ytest_df, closing_price_df, share_amount, starting_value, lookahead, save_recent=False):
    long_short_order_book = add_long_short_shares(ytest_df['bs_signal'], share_amount)
    buy_sell_order_book = add_buy_sell_shares(ytest_df['bs_signal'], closing_price_df, starting_value)
    long_short_portfolio_values = compute_portfolio(long_short_order_book, closing_price_df)
    buy_sell_portfolio_values = compute_portfolio(buy_sell_order_book, closing_price_df)
    # buy_sell_portfolio_values.to_csv('best_case_buy.csv')
    # long_short_portfolio_values.to_csv('best_case_long.csv')
    if save_recent:
        graph_order_book(buy_sell_portfolio_values, closing_price_df, 'best_case', 'buy_sell',
                         'no indicators used', lookahead)
        graph_order_book(long_short_portfolio_values, closing_price_df, 'best_case', 'long_short',
                         'no indicators used', lookahead)
    long_short_yearly_gains_dict, long_short_total_percent_gain = compute_yearly_gains(long_short_portfolio_values)
    buy_sell_yearly_gains_dict, buy_sell_total_percent_gain = compute_yearly_gains(buy_sell_portfolio_values)
    output = [f'best long_short percent gain {long_short_total_percent_gain}, best buy sell percent gain {buy_sell_total_percent_gain}',
              ]

    OUTPUT.extend(output)
    return (long_short_total_percent_gain, long_short_yearly_gains_dict), (buy_sell_total_percent_gain, buy_sell_yearly_gains_dict)

def compute_portfolio(order_book, closing_price_df, commission=9.95, impact=0.005):
    order_book_copy = order_book.copy()
    filled_orders = pd.DataFrame(index=order_book_copy.index, columns=['holding', 'cash_earned', 'bankroll', 'cumulative_percentage'])
    shares_holder = 0
    gains_holder = 0
    initial_spending = 0
    start_date = order_book_copy.index[0]
    for index in order_book_copy.index:
        if order_book_copy.loc[index, 'bs_signal'] == BUY:
            shares_holder += order_book_copy.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((closing_price_df.loc[index, 'Close'] * abs(order_book_copy.loc[index, 'share_amount']) * (1.000 + impact)) + commission)
            filled_orders['cash_earned'][index] = -cost
            if index == start_date:
                gains_holder = 0
                filled_orders['bankroll'][index] = gains_holder
                filled_orders['cumulative_percentage'][index] = 0
            else:
                gains_holder -= cost
                filled_orders['bankroll'][index] = gains_holder
                filled_orders['cumulative_percentage'][index] = ((filled_orders['bankroll'][index] - abs(filled_orders['cash_earned'][start_date])) / abs(filled_orders['cash_earned'][start_date])) * 100

        elif order_book_copy.loc[index, 'bs_signal'] == SELL:
            shares_holder -= order_book_copy.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((closing_price_df.loc[index, 'Close'] * abs(order_book_copy.loc[index, 'share_amount']) * (1.000 - impact)) - commission)
            filled_orders['cash_earned'][index] = cost
            gains_holder += cost
            filled_orders['bankroll'][index] = gains_holder
            filled_orders['cumulative_percentage'][index] = ((filled_orders['bankroll'][index] - abs(filled_orders['cash_earned'][start_date])) / abs(filled_orders['cash_earned'][start_date])) * 100
    order_book_copy['cash_earned'] = filled_orders['cash_earned']
    order_book_copy['holding'] = filled_orders['holding']
    order_book_copy['bankroll'] = filled_orders['bankroll']
    order_book_copy['cumulative_percentage'] = filled_orders['cumulative_percentage']
    order_book_copy['close'] = closing_price_df['Close']
    order_book_copy = order_book_copy.dropna()
    OUTPUT.append('model net profits {} from {} to {}'.format(order_book_copy.sum()['cash_earned'], order_book_copy.index[0], order_book_copy.index[-1]))
    return order_book_copy


def compute_simple_baseline(name, close_df, share_amount, commission=9.95, impact=0.005):
    """
    Computes the value of holding the share amnt on the first day
    INPUT: BUY/SELL orderbook DO NOT use LONG/SHORTS
    """
    close_df_copy = close_df.copy()
    close_df_copy = close_df_copy.dropna()
    start_date = close_df_copy.index[0]
    end_date = close_df_copy.index[-1]
    filled_orders = pd.DataFrame(index=close_df_copy.index, columns=['holding', 'cash_earned', 'bankroll', 'cumulative_percent'])
    shares_holder = 0
    gains_holder = 0

    # Execute buy
    shares_holder += share_amount
    filled_orders['holding'][start_date] = shares_holder
    cost = float((close_df_copy['Close'][0] * share_amount * (1.000 + impact)) + commission)
    filled_orders['cash_earned'][start_date] = -cost
    gains_holder -= cost
    filled_orders['bankroll'][start_date] = gains_holder
    filled_orders['cumulative_percent'][start_date] = (gains_holder - filled_orders['bankroll'][start_date])/abs(filled_orders['bankroll'][start_date]) * 100
    # Execute sell
    shares_holder -= share_amount
    filled_orders['holding'][end_date] = shares_holder
    cost = float((close_df_copy['Close'][-1] * share_amount * (1.000 + impact)) + commission)
    filled_orders['cash_earned'][end_date] = cost
    gains_holder += cost
    filled_orders['bankroll'][end_date] = gains_holder
    filled_orders['cumulative_percent'][end_date] = (gains_holder - filled_orders['bankroll'][start_date])/abs(filled_orders['bankroll'][start_date]) * 100

    summed = filled_orders.sum()
    filled_orders = filled_orders.dropna()
    percent_increase = (close_df_copy['Close'][-1] - close_df_copy['Close'][0]) / abs(close_df_copy['Close'][0]) * 100
    output = [
        'If you bought {} shares of {} on {}, you would have {} on {}'.format(filled_orders['holding'][0], name, start_date, summed['cash_earned'], end_date),
        'The share increased by {} %'.format(percent_increase)
        ]
    OUTPUT.extend(output)
    return filled_orders


def compute_yearly_gains(order_book):
    yearly_gains_dict = {}
    order_book_start = datetime.strptime(order_book.index[0], '%Y-%m-%d')
    order_book_end = datetime.strptime(order_book.index[-1], '%Y-%m-%d')
    x = 1
    date_holder = '{}-{}-{}'.format(order_book_start.year, order_book_start.month, order_book_start.day)
    next_years_date = '{}-{}-{}'.format(order_book_start.year + x, order_book_start.month, order_book_start.day)
    while int(next_years_date.split('-')[0]) <= int(order_book_end.year):
        if x == 1:
            yearly_gain = ((order_book.loc[:next_years_date, 'bankroll'][-1] - abs(order_book.loc[date_holder:, 'cash_earned'][
                0])) / abs(order_book.loc[date_holder:, 'cash_earned'][0]) * 100)
        else:
            yearly_gain = ((order_book.loc[:next_years_date, 'bankroll'][-1] - order_book.loc[date_holder:, 'bankroll'][0])/abs(order_book.loc[date_holder:, 'bankroll'][0]) * 100)
        OUTPUT.append('From {} to {} the yearly gain is {}'.format(date_holder, next_years_date, yearly_gain))
        date_holder = next_years_date
        yearly_gains_dict[x] = yearly_gain
        x += 1
        next_years_date = '{}-{}-{}'.format(order_book_start.year + x, order_book_start.month, order_book_start.day)

    order_book_end_str = '{}-{}-{}'.format(order_book_end.year, order_book_end.month, order_book_end.day)
    remaining_gains = order_book.loc[date_holder:order_book_end_str].sum()['cash_earned']
    if remaining_gains > 0.0:
        yearly_gain = ((order_book.loc[:order_book_end_str, 'bankroll'][-1] - order_book.loc[date_holder:, 'bankroll'][0])/abs(order_book.loc[date_holder:, 'bankroll'][0]) * 100)
        yearly_gains_dict[x] = yearly_gain
    if order_book['bs_signal'][-1] == BUY:
        total_percent_gain = order_book['cumulative_percentage'][-2]
    else:
        total_percent_gain = order_book['cumulative_percentage'][-1]

    OUTPUT.append('THE TOTAL % gain : {}'.format(total_percent_gain))
    return yearly_gains_dict, total_percent_gain

def get_spy(spy_close_df, args):
    spy_hold_portfolio = compute_simple_baseline("spy", spy_close_df, args['share_amount'])
    graph_spy(spy_hold_portfolio, spy_close_df)
    print('\n'.join(OUTPUT))


def compare_strategies(buy_sell_order_book, long_short_order_book, target_close_df, args):
    OUTPUT.append('generating longs and shorts ')
    long_short_portfolio_values = compute_portfolio(long_short_order_book, target_close_df)
    long_short_yearly_gains_dict, long_short_total_percent_gain = compute_yearly_gains(long_short_portfolio_values)
    OUTPUT.append('generating buy and sell')
    buy_sell_portfolio_values = compute_portfolio(buy_sell_order_book, target_close_df)
    buy_sell_yearly_gains_dict, buy_sell_total_percent_gain = compute_yearly_gains(buy_sell_portfolio_values)

    OUTPUT.append('generating baselines')
    target_hold_portfolio = compute_simple_baseline(args["file_name"], target_close_df, int(args['share_amount']))
    if args["inspect"]:
        print('\n'.join(OUTPUT))
    if args['save_recent']:
        long_short_portfolio_values.to_csv(os.path.join(RECENT_RUN_DIR, 'long_short_portfolio.csv'))
        buy_sell_portfolio_values.to_csv(os.path.join(RECENT_RUN_DIR, 'buy_short_portfolio.csv'))
        graph_order_book(long_short_portfolio_values, target_close_df, args['model_name'], args['file_name'],
                         args["indicators"], args['length'])

    return buy_sell_portfolio_values, buy_sell_total_percent_gain, long_short_portfolio_values, long_short_total_percent_gain


def graph_spy(spy_order_book, spy_close):
    plt.plot(list(spy_close.index), spy_close['Close'], label='price')
    plt.plot(spy_order_book.index[0], spy_close['Close'][0], 's', label='buy')
    plt.plot(spy_order_book.index[1], spy_close['Close'][-1], 's', label='sell')
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
    plt.plot(sell_markers_date, sell_markers_price, 's', label='sell')
    plt.plot(buy_markers_date, buy_markers_price, 's', label='buy')

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
    plt.savefig('{}.png'.format(os.path.join(RECENT_RUN_DIR, title)), dpi=300)
    plt.figure().clear()
