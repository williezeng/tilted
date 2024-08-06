import backtrader as bt
from backtrader.feeds import PandasData


class AlphaStrategy(bt.Strategy):
    params = (
        ('equity_pct', 0.9),
        ('stop_loss_pct', 0.05),
        ('printlog', False),
    )

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.bs_signal = self.datas[0].openinterest

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.highest_price = None

    def log(self, txt, dt=None):
        """ Logging function for this strategy"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.highest_price = self.buyprice  # Initialize highest price to the buy price
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
                self.buyprice = None
                self.highest_price = None  # Reset highest price after selling
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATING PROFIT, GROSS: %.2f, NET: %.2f' % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Check if there's an order pending. If yes, we cannot issue another.
        if self.order:
            return
        available_cash = self.broker.get_cash()

        # Update highest price if we have a position
        if self.position and self.dataclose[0] > self.highest_price:
            self.highest_price = self.dataclose[0]

        # if we're n the market check for stop-loss condition
        if self.position:
            stop_loss_price = self.highest_price * (1.0 - self.params.stop_loss_pct)
            if self.dataclose[0] <= stop_loss_price:
                self.log('STOP-LOSS TRIGGERED, SELLING AT %.2f' % self.dataclose[0])
                self.order = self.sell(size=self.position.size)
            # sell if we have signal
            elif self.bs_signal[0] == -1:
                self.order = self.sell(size=self.position.size)

        # buy if we are not in the market
        else:
            if self.bs_signal[0] == 1:
                position_size = available_cash * self.params.equity_pct
                self.order = self.buy(size=position_size / self.data.close[0])

