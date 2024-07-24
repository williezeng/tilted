import os
import pandas as pd
import numpy as np
import matplotlib
import traceback
from datetime import datetime
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
from utils import trading_logger
import matplotlib.pyplot as plt
from utils.constants import BUY, SELL, HOLD

RECENT_RUN_DIR = os.path.join(os.path.curdir, 'recent_run')
logger = trading_logger.getlogger()
OUTPUT = []


def analyze_prediction_to_test(ypred, ytest, dataframe_from_tickers, share_amount, starting_value, lookahead_days, save_recent):
    check_buy_sell_signals(ypred, ytest)
    output = [f"\nHere is the best case scenario:"]
    OUTPUT.extend(output)

    buy_sell_order_book = add_buy_sell_shares(ytest['bs_signal'], dataframe_from_tickers, starting_value)

    compute_portfolio(buy_sell_order_book, dataframe_from_tickers)
    print(OUTPUT)

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

#
# def compute_best_case(ytest_df, closing_price_df, share_amount, starting_value, lookahead, save_recent=False):
#     buy_sell_order_book = add_buy_sell_shares(ytest_df['bs_signal'], closing_price_df, starting_value)
#     buy_sell_portfolio_values, benchmark_df = compute_portfolio(buy_sell_order_book, closing_price_df)
#     # buy_sell_portfolio_values.to_csv('best_case_buy.csv')
#     if save_recent:
#         graph_order_book(buy_sell_portfolio_values, closing_price_df, 'best_case', 'buy_sell', 'no indicators used', lookahead)
#     buy_sell_total_percent_gain_df = compute_yearly_gains(buy_sell_portfolio_values)


def compute_portfolio(order_book, closing_price_df, commission=9.95, impact=0.005):
    order_book_copy = order_book.copy()
    filled_orders = pd.DataFrame(index=order_book_copy.index, columns=['close', 'holding', 'executed_cost', 'bankroll', 'total_portfolio_value', 'cumulative_percentage'])
    benchmark_df = pd.DataFrame(index=order_book_copy.index, columns=['close', 'holding', 'baseline_portfolio_value', 'baseline_percentage'])
    shares_holder = 0
    gains_holder = 0

    start_date = order_book_copy.index[0]
    start_amount = order_book_copy.loc[start_date, 'share_amount']
    for index in order_book_copy.index:
        close_price = closing_price_df['Close'][index]
        filled_orders['close'][index] = close_price
        benchmark_df['close'][index] = close_price
        total_share_value = close_price * order_book_copy.loc[index, 'share_amount']
        benchmark_df['holding'][index] = start_amount
        benchmark_df['baseline_portfolio_value'][index] = close_price * start_amount
        benchmark_df['baseline_percentage'][index] = ((benchmark_df['baseline_portfolio_value'][index] -
                                                             benchmark_df['baseline_portfolio_value'][start_date]) /
                                                            benchmark_df['baseline_portfolio_value'][start_date]) * 100

        if order_book_copy.loc[index, 'bs_signal'] == BUY:
            shares_holder += order_book_copy.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((total_share_value * (1.000 + impact)) + commission)
            if index == start_date:
                gains_holder = 0
                filled_orders['bankroll'][index] = gains_holder
                filled_orders['total_portfolio_value'][index] = abs(total_share_value)
                filled_orders['cumulative_percentage'][index] = 0
                benchmark_df['baseline_portfolio_value'][index] = abs(total_share_value)
                benchmark_df['baseline_percentage'][index] = 0
            else:
                gains_holder -= cost
                filled_orders['bankroll'][index] = gains_holder
                filled_orders['total_portfolio_value'][index] = abs(total_share_value) + filled_orders['bankroll'][index]
                filled_orders['cumulative_percentage'][index] = ((filled_orders['total_portfolio_value'][index] - filled_orders['total_portfolio_value'][start_date]) /
                                                                 filled_orders['total_portfolio_value'][start_date]) * 100

        elif order_book_copy.loc[index, 'bs_signal'] == SELL:
            filled_orders['close'][index] = closing_price_df['Close'][index]
            shares_holder -= order_book_copy.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_price * abs(order_book_copy.loc[index, 'share_amount']) * (1.000 - impact)) - commission)
            filled_orders['executed_cost'][index] = cost
            gains_holder += cost
            filled_orders['bankroll'][index] = gains_holder
            filled_orders['total_portfolio_value'][index] = filled_orders['bankroll'][index]
            filled_orders['cumulative_percentage'][index] = ((filled_orders['total_portfolio_value'][index] - filled_orders['total_portfolio_value'][start_date]) /
                                                             filled_orders['total_portfolio_value'][start_date]) * 100

    portfolio_df = pd.concat([order_book_copy, filled_orders, ], axis=1)
    OUTPUT.append(f"model net profits {portfolio_df['cumulative_percentage'][-1]} % from {order_book_copy.index[0]} to {order_book_copy.index[-1]}\n"
                  f"The baseline {benchmark_df['baseline_percentage'][-1]} % from {order_book_copy.index[0]} to {order_book_copy.index[-1]}\n")
    return portfolio_df, benchmark_df


def compute_yearly_gains(order_book):
    end_of_year_portfolio_df = order_book.groupby(pd.to_datetime(order_book.index).year).last()
    return end_of_year_portfolio_df



def compare_strategies(yprediction, target_close_df, file_name, model_name, indicators, length, share_amount, starting_value, inspect, save_recent):
    # OUTPUT.append('generating longs and shorts ')
    # long_short_portfolio_values = compute_portfolio(long_short_order_book, target_close_df)
    # long_short_yearly_gains_dict, long_short_total_percent_gain = compute_yearly_gains(long_short_portfolio_values)
    buy_sell_order_book = add_buy_sell_shares(yprediction, target_close_df, starting_value)
    OUTPUT.append('_ Generating Summary _')
    buy_sell_portfolio_values, benchmark_df = compute_portfolio(buy_sell_order_book, target_close_df)
    yearly_end_of_year_df = compute_yearly_gains(buy_sell_portfolio_values)
    if inspect:
        print('\n'.join(OUTPUT))
    if save_recent:
        # long_short_portfolio_values.to_csv(os.path.join(RECENT_RUN_DIR, 'long_short_portfolio.csv'))
        buy_sell_portfolio_values.to_csv(os.path.join(RECENT_RUN_DIR, 'buy_short_portfolio.csv'))
        graph_order_book(buy_sell_portfolio_values, target_close_df, model_name, file_name,
                         indicators, length)

    return buy_sell_portfolio_values, yearly_end_of_year_df['cumulative_percentage'].iloc[-1]

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



