

def check_buy_sell_signals(model_object):
    correct_buys = 0
    correct_sells = 0
    total_len = len(model_object.ypred)
    total_buys = len([x for x in model_object.ytest['bs_signal'] if x == 1])
    total_sells = len([x for x in model_object.ytest['bs_signal'] if x == -1])
    for x in range(total_len):
        if model_object.ypred[x] == model_object.ytest['bs_signal'][x]:
            if model_object.ypred[x] == 1:  # buy
                correct_buys += 1
            elif model_object.ypred[x] == -1:  # sell
                correct_sells += 1

    percentage_buys = (correct_buys/total_buys) * 100
    percentage_sells = (correct_sells/total_sells) * 100
    print('---------------------')
    print('there are supposed to be {} buys. there are supposed to be {} sells'.format(total_buys, total_sells))
    print('you made {} correct buys. you made {} correct sells'.format(correct_buys, correct_sells))
    print('correct buy percentage {} and correct sell percentage {}'.format(percentage_buys, percentage_sells))
