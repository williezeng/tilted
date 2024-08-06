# For details on your rights to use the actual data downloaded. Remember - the Yahoo! finance API is intended for personal use only.
import sys
import os
import shutil
import pandas as pd
import datetime
from pandas_datareader import data as pdr
import multiprocessing
import yfinance as yf
import argparse
from utils import constants

DATA_DIR = os.path.join(os.path.curdir, 'yahoo_data')


def get_fortune_500_tickers():
    tickers = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return tickers['Symbol']


def fetch_data(name, start, end):
    assert isinstance(name, str), '%s must be a str'.format(name)
    assert isinstance(start, str), '%s must be a str'.format(start)
    assert isinstance(end, str), '%s must be a str'.format(end)
    def validate(date_text):
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
    validate(start)
    validate(end)
    dataframe = yf.download(name, start=start, end=end)
    return dataframe


def build_args():
    parser = argparse.ArgumentParser()
    today = datetime.date.today()
    parser.add_argument('--start', help='the start date of the stock', type=str, default="2008-01-01")
    parser.add_argument('--end', help='the end date of the stock', type=str, default=today.strftime("%Y-%m-%d"))
    parser.add_argument('--name', help='the name of the stock', type=str, required=False)
    parser.add_argument('--top500', help=f'get the top500 tickers', type=bool, required=False, default=False)
    parser.add_argument('--all', help=f'get all tickers in ', action='store_true', default=False)

    return parser.parse_args()


def write_to_file(params):
    name, start_date, end_date = params
    df = fetch_data(name, start_date, end_date)
    df.to_parquet(os.path.join(DATA_DIR, '{}.parquet'.format(name)))


def convert_ticker(ticker):
    return ticker.replace('.', '-')


def gather_all_fortune500(start, end):
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    fortune_df = get_fortune_500_tickers()
    tickers_list = fortune_df.tolist()
    converted_tickers = [convert_ticker(ticker) for ticker in tickers_list]
    params_list = [(ticker, start, end) for ticker in converted_tickers if
                   ticker not in constants.BANNED_TICKERS]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(write_to_file, params_list)


if __name__ == "__main__":
    args = build_args()
    if args.top500:
        fortune_500_df = get_fortune_500_tickers()
        fortune_500_df.to_csv(os.path.join(DATA_DIR, '00_fortune_500_tickers.csv'), index=False, header=False)
    elif args.all:
        gather_all_fortune500(args.start, args.end)
    elif args.name:
        params = (args.name, args.start, args.end)
        write_to_file(params)
    else:
        exit('need either --name or --all_in_folder')


