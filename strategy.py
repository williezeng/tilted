import backtrader as bt


class AlphaStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        """ Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.datasignal = self.datas[0].openinterest

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
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
        # buy
        if self.datasignal[0] == 1:
            position_size = available_cash * self.params.equity_pct
            if not self.position:  # buy if we are not in the market
                self.order = self.buy(size=position_size / self.data.close[0])
        # sell
        elif self.datasignal[0] == -1:
            if self.position:  # sell if we are in the market
                self.order = self.sell(size=self.position.size)

