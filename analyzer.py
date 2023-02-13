import pandas as pd

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

def compute_portvals(book_order, close_data, start_val=1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code

    # NOTE: orders_file may be a string, or it may be a file object. Your

    # code should work correctly with either input


    sorted_book_order = book_order.sort_index()
    start_date = sorted_book_order.index[0]
    end_date = sorted_book_order.index[-1]
    import pdb
    pdb.set_trace()
    unique_symbol_list = sorted_book_order['Symbol'].unique().tolist()

    dates = pd.date_range(start_date, end_date, name='Date', freq='D')
    # the prices need to be front filled and back filled
    # ffilled_prices = full_date_prices.join(adjusted_close_all)
    # ffilled_prices.fillna(method="ffill", inplace=True)
    # ffilled_prices.fillna(method="bfill", inplace=True)
    # share_cash_tracker = pd.DataFrame(index=ffilled_prices.index, columns=ffilled_prices.columns)
    # share_cash_tracker['Cash'] = 0
    # share_cash_tracker[:] = 0.0
    for index, col_value in sorted_book_order.iterrows():
        if col_value['Order'] == 'BUY' and col_value['Shares']:
            share_cash_tracker.at[index, col_value['Symbol']] += col_value['Shares']
            cost = float(
                commission + ffilled_prices.at[index, col_value['Symbol']] * (1.000 + impact) * col_value['Shares'])
            share_cash_tracker.at[index, 'Cash'] -= cost
        else:
            share_cash_tracker.at[index, col_value['Symbol']] -= col_value['Shares']
            cost = float(
                ffilled_prices.at[index, col_value['Symbol']] * (1.000 - impact) * col_value['Shares'] - commission)
            share_cash_tracker.at[index, 'Cash'] += cost


    holdings = share_cash_tracker.copy()

    holdings.at[start_date, 'Cash'] = holdings.at[start_date, 'Cash'] + start_val

    holding_orders = holdings.cumsum(axis=0)
    if holding_orders['SPY'].sum(axis=0) == 0:
        holding_orders = holding_orders.drop(['SPY'], axis=1)
        adjusted_close_all = adjusted_close_all.drop(['SPY'], axis=1)
    adjusted_close_all['Cash'] = 1

    values = adjusted_close_all * holding_orders
    values = values.dropna()
    values['Sum'] = values.sum(axis=1)
    portvals = values.drop(values.columns[:-1], axis=1)

    return portvals