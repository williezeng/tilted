import pandas as pd


def get_technical_indicators_and_buy_sell_dfs(file_path):
    technical_indicators_df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
    buy_sell_signal_df = technical_indicators_df.pop('bs_signal')
    return technical_indicators_df, buy_sell_signal_df
