import unittest
import os
from utils import constants
import pandas as pd
from analyzer import compute_portvals, graph_order_book, compute_yearly_gains
import tech_indicators
from datetime import datetime

class TestMarketSimETH(unittest.TestCase):
    def setUp(self):
        file_name = os.path.join(constants.YAHOO_DATA_DIR, 'real_eth.csv')
        self.data_frame_from_file = tech_indicators.read_df_from_file(file_name)
        self.long_short_order_bk = pd.read_csv('tester1.csv', index_col=[0], header=[0], skipinitialspace=True)

    def test_create_long_short_portfolio(self):
        portfolio = compute_portvals(self.long_short_order_bk, self.data_frame_from_file[['Close']], 0)
        sum_series = portfolio.sum()
        self.assertEqual(sum_series['share_amount'], 18900.0)
        self.assertEqual(sum_series['bs_signal'], 1)
        self.assertEqual(sum_series['cash_earned'], 938395.1612033314)
        self.assertEqual(sum_series['holding'], 100.0)
        self.assertEqual(sum_series['close'], 115489.94024682)
        import pdb
        pdb.set_trace()

    def test_compute_yearly_gains(self):
        portfolio = compute_portvals(self.long_short_order_bk, self.data_frame_from_file[['Close']], 0)
        yearly_gains_dict = compute_yearly_gains(portfolio)
        self.assertEqual(yearly_gains_dict[1], 564.3000105961332)
        self.assertEqual(yearly_gains_dict[2], -100.08058630201029)
        self.assertEqual(yearly_gains_dict[3], 136.26685212517867)
        self.assertEqual(yearly_gains_dict[4], 316.76851649782077)
        self.assertEqual(yearly_gains_dict[5], 154.6028791479902)

        first = portfolio.loc['2017-12-10':'2018-12-10']['cash_earned'].sum()
        second = portfolio.loc['2018-12-10':'2019-12-10']['cash_earned'].sum()
        third = portfolio.loc['2019-12-10':'2020-12-10']['cash_earned'].sum()
        fourth = portfolio.loc['2020-12-10':'2021-12-10']['cash_earned'].sum()
        fifth = portfolio.loc['2021-12-10':'2022-12-10']['cash_earned'].sum()

        total_percent_gain = ((portfolio['cash_earned'].sum() - portfolio['cash_earned'][0])/abs(portfolio['cash_earned'][0])) * 100

        import pdb
        pdb.set_trace()


    def test_create_ylabels(self):
        df_with_ytest = tech_indicators.create_ylabels(self.data_frame_from_file, lookahead_days=5)


