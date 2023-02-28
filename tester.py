import unittest
import os

import pandas

from utils import constants
import pandas as pd
from analyzer import compute_portfolio, graph_order_book, compute_yearly_gains, compute_best_case, compute_simple_baseline
import tech_indicators
from datetime import datetime

LONG_SHORT_TEST_FILE = 'long_short_tester.csv'
BUY_SELL_TEST_FILE = 'buy_sell_tester.csv'
YTEST_TEST_FILE = 'ytest.csv'
Y_PRED_TEST_FILE = 'ypred.csv'

class TestMarketSimETH(unittest.TestCase):
    def setUp(self):
        file_name = os.path.join(constants.YAHOO_DATA_DIR, 'real_eth.csv')
        self.data_frame_from_file = tech_indicators.read_df_from_file(file_name)
        self.long_short_order_bk = pd.read_csv(LONG_SHORT_TEST_FILE, index_col=[0], header=[0], skipinitialspace=True)
        self.buy_sell_order_bk = pd.read_csv(BUY_SELL_TEST_FILE, index_col=[0], header=[0], skipinitialspace=True)
        self.ytest = pd.read_csv(YTEST_TEST_FILE, index_col=[0], header=[0], skipinitialspace=True)
        self.ypred = pd.read_csv(Y_PRED_TEST_FILE, index_col=[0], header=[0], skipinitialspace=True)
        self.assertIsInstance(self.ytest, pandas.DataFrame)
        self.assertIsInstance(self.ypred, pandas.DataFrame)

    def test_create_long_short_orderbook(self):
        long_short_order_book = tech_indicators.add_long_short_shares(self.ypred['bs_signal'], 15)
        self.assertIsInstance(long_short_order_book, pandas.DataFrame)

        self.assertEqual(long_short_order_book['share_amount']['2022-10-05'], 30)
        self.assertEqual(long_short_order_book['bs_signal']['2022-10-05'], -1)

        self.assertEqual(long_short_order_book['share_amount']['2022-09-27'], 30)
        self.assertEqual(long_short_order_book['bs_signal']['2022-09-27'], 1)

        self.assertEqual(long_short_order_book.sum()['share_amount'], 3465)

    def test_create_buy_sell_orderbook(self):
        buy_sell_order_book = tech_indicators.add_buy_sell_shares(self.ypred['bs_signal'], self.data_frame_from_file[['Close']], 1200)
        self.assertIsInstance(buy_sell_order_book, pandas.DataFrame)

        self.assertEqual(buy_sell_order_book['share_amount']['2022-10-05'], 0.8745859212461541)
        self.assertEqual(buy_sell_order_book['bs_signal']['2022-10-05'], -1)

        self.assertEqual(buy_sell_order_book['share_amount']['2022-09-27'], 0.8745859212461541)
        self.assertEqual(buy_sell_order_book['bs_signal']['2022-09-27'], 1)

        self.assertEqual(buy_sell_order_book.sum()['share_amount'], 440.1801094115214)

    def test_compute_long_short_portfolio(self):
        # share_amount = 15
        portfolio = compute_portfolio(self.long_short_order_bk, self.data_frame_from_file[['Close']])
        # verify the sum to ensure all rows are the same value
        sum_series = portfolio.sum()
        self.assertEqual(sum_series['share_amount'], 3885)
        self.assertEqual(sum_series['bs_signal'], 0)
        self.assertEqual(sum_series['cash_earned'], 140780.9731574098)
        self.assertEqual(sum_series['holding'], 0)
        self.assertEqual(sum_series['close'], 150100.07869012)
        self.assertEqual(sum_series['cumulative_percentage'], 79961.0437579257)
        self.assertEqual(sum_series['bankroll'], 6268300.377993768)

    def test_compute_buy_sell_portfolio(self):
        # starting_value = 1200
        portfolio = compute_portfolio(self.buy_sell_order_bk, self.data_frame_from_file[['Close']])
        self.assertIsInstance(portfolio, pandas.DataFrame)
        sum_series = portfolio.sum()
        self.assertEqual(sum_series['share_amount'], 356.84805909715647)
        self.assertEqual(sum_series['bs_signal'], 0)
        self.assertEqual(sum_series['cash_earned'], 9979.820684835644)
        self.assertEqual(sum_series['holding'], 178.42402954857826)
        self.assertEqual(sum_series['close'], 150100.07869012)
        self.assertEqual(sum_series['cumulative_percentage'], 38775.480636958426)
        self.assertEqual(sum_series['bankroll'], 566027.3771569243)

    def test_compute_yearly_gains(self):
        portfolio = compute_portfolio(self.long_short_order_bk, self.data_frame_from_file[['Close']], 0)
        self.assertIsInstance(portfolio, pandas.DataFrame)
        yearly_gains_dict, total_percent_gains = compute_yearly_gains(portfolio)
        self.assertEqual(yearly_gains_dict[1], 170.018781953317)
        self.assertEqual(yearly_gains_dict[2], 13.406855604374336)
        self.assertEqual(yearly_gains_dict[3], -16.59490112422985)
        self.assertEqual(yearly_gains_dict[4], 301.05179026447547)
        self.assertEqual(yearly_gains_dict[5], 262.06312764315476)
        self.assertEqual(total_percent_gains, 2107.8536368928735)

    def test_compute_baseline(self):
        target_hold_portfolio = compute_simple_baseline(self.data_frame_from_file[['Close']], 15)
        self.assertIsInstance(target_hold_portfolio, pandas.DataFrame)
        self.assertEqual(target_hold_portfolio['cash_earned']['2017-11-09'], -4847.2763407024995)
        self.assertEqual(target_hold_portfolio['cash_earned']['2022-12-20'], 18263.472896749998)


    def test_compute_best_case(self):
        long_short_buy_sell_tup = compute_best_case(self.ytest, self.data_frame_from_file[['Close']], 15, 1200)
        long_short_gain = 6142.996966219385
        long_short_yearly_gain = {1: 982.3363534138012, 2: 20.906795951234695, 3: 3.493270942019021,
                                  4: 260.1765416855101, 5: 85.40608464578555, 6: 12.508897362931776}


        buy_sell_gain = 15266.478263044957
        buy_sell_yearly_gain = {1: 668.1219090406433, 2: 142.85521377461805, 3: 69.32580646180146,
                                4: 423.3464648335443, 5: 196.82112458338483, 6: 43.66835364435849}
        self.assertEqual(long_short_buy_sell_tup[0][0], long_short_gain)
        self.assertEqual(long_short_buy_sell_tup[0][1], long_short_yearly_gain)

        self.assertEqual(long_short_buy_sell_tup[1][0], buy_sell_gain)
        self.assertEqual(long_short_buy_sell_tup[1][1], buy_sell_yearly_gain)


if __name__ == '__main__':
    unittest.main()
