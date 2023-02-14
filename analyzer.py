import pandas as pd
import numpy as np
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

def compute_portvals(book_order, close_data, start_val=100000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code

    # NOTE: orders_file may be a string, or it may be a file object. Your

    # code should work correctly with either input


    start_date = book_order.index[0]
    end_date = book_order.index[-1]
    # start_index = list(close_data.index).index(start_date)
    # end_index = list(close_data.index).index(end_date)
    # close_data = close_data[start_index:end_index]
    # book_order['close'] = close_data
    dates = pd.date_range(start_date, end_date, name='Date', freq='D')
    # the prices need to be front filled and back filled
    # ffilled_prices = full_date_prices.join(adjusted_close_all)
    # ffilled_prices.fillna(method="ffill", inplace=True)
    # ffilled_prices.fillna(method="bfill", inplace=True)
    # share_cash_tracker = pd.DataFrame(index=ffilled_prices.index, columns=ffilled_prices.columns)
    # share_cash_tracker['Cash'] = 0
    # share_cash_tracker[:] = 0.0

    filled_orders = pd.DataFrame(index=book_order.index, columns=['holding', 'total_liquid_cash'])
    filled_orders['total_liquid_cash'][0] = start_val
    shares_holder = 0

    for index in book_order.index:
        if book_order.loc[index, 'bs_signal'] == BUY and not np.isnan(book_order.loc[index, 'share_amount']):
            shares_holder += book_order.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_data.loc[index, 'Close'] * abs(book_order.loc[index, 'share_amount']) * (1.000 + impact)) + commission)
            filled_orders['total_liquid_cash'][index] = cost
        elif book_order.loc[index, 'bs_signal'] == SELL and not np.isnan(book_order.loc[index, 'share_amount']):       # SELL
            shares_holder -= book_order.loc[index, 'share_amount']
            filled_orders['holding'][index] = shares_holder
            cost = float((close_data.loc[index, 'Close'] * abs(book_order.loc[index, 'share_amount']) * (1.000 - impact)) - commission)
            filled_orders['total_liquid_cash'][index] = cost
    book_order['total_liquid_cash'] = filled_orders['total_liquid_cash']
    book_order['holding'] = filled_orders['holding']
    book_order['close'] = close_data['Close']

    import pdb
    pdb.set_trace()
    #
    # holdings = share_cash_tracker.copy()
    # essentially buy the amnt of shares
    # holdings.at[start_date, 'Cash'] = holdings.at[start_date, 'Cash'] + start_val
    #
    # holding_orders = holdings.cumsum(axis=0)
    # if holding_orders['SPY'].sum(axis=0) == 0:
    #     holding_orders = holding_orders.drop(['SPY'], axis=1)
    #     adjusted_close_all = adjusted_close_all.drop(['SPY'], axis=1)
    # adjusted_close_all['Cash'] = 1
    #
    # values = adjusted_close_all * holding_orders
    # values = values.dropna()
    # values['Sum'] = values.sum(axis=1)
    # portvals = values.drop(values.columns[:-1], axis=1)
    #
    # return portvals