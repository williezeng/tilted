import unittest
import os
from utils import constants
import pandas as pd
from analyzer import compute_portvals, graph_order_book
import tech_indicators


class TestMarketSimETH(unittest.TestCase):
    def setUp(self):
        file_name = os.path.join(constants.YAHOO_DATA_DIR, 'real_eth.csv')
        self.data_frame_from_file = tech_indicators.read_df_from_file(file_name)
        self.order_bk = pd.read_csv('tester1.csv', index_col=[0], header=[0], skipinitialspace=True)

    def test_create_correct_order_book(self):
        order_book = compute_portvals(self.order_bk, self.data_frame_from_file[['Close']])
        sum_series = order_book.sum()
        self.assertEqual(sum_series['share_amount'], 18900.0)
        self.assertEqual(sum_series['bs_signal'], 1)
        self.assertEqual(sum_series['total_liquid_cash'], 938395.1612033314)
        self.assertEqual(sum_series['holding'], 100.0)
        self.assertEqual(sum_series['close'], 115489.94024682)

