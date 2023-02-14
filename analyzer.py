import pandas as pd
import numpy as np
import matplotlib
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

def compute_portvals(order_book, close_data, start_val=100000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code

    # NOTE: orders_file may be a string, or it may be a file object. Your

    # code should work correctly with either input


    start_date = order_book.index[0]
    end_date = order_book.index[-1]
    # start_index = list(close_data.index).index(start_date)
    # end_index = list(close_data.index).index(end_date)
    # close_data = close_data[start_index:end_index]
    # book_order['close'] = close_data
    filled_orders = pd.DataFrame(index=order_book.index, columns=['holding', 'total_liquid_cash'])
    filled_orders['total_liquid_cash'][0] = start_val
    shares_holder = 0

    for index in order_book.index:
        if order_book.loc[index, 'bs_signal'] == BUY and not np.isnan(order_book.loc[index, 'share_amount']):
            shares_holder += order_book.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_data.loc[index, 'Close'] * abs(order_book.loc[index, 'share_amount']) * (1.000 + impact)) + commission)
            filled_orders['total_liquid_cash'][index] = -cost
        elif order_book.loc[index, 'bs_signal'] == SELL and not np.isnan(order_book.loc[index, 'share_amount']):       # SELL
            shares_holder -= order_book.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_data.loc[index, 'Close'] * abs(order_book.loc[index, 'share_amount']) * (1.000 - impact)) - commission)
            filled_orders['total_liquid_cash'][index] = cost
    order_book['total_liquid_cash'] = filled_orders['total_liquid_cash']
    order_book['holding'] = filled_orders['holding']
    order_book['close'] = close_data['Close']
    order_book = order_book.dropna()
    print(order_book.sum())
    return order_book

def graph_order_book(order_book, close_data, model_name, file_name, list_of_indicators, moving_average_length):
    # figure = plt.figure()
    # ax = close_data.plot(y='Close', x='Date')
    import matplotlib.dates as mdates
    matplotlib.use('TkAgg')
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


def compare_strategies(generated_order_book, baseline):
    pass