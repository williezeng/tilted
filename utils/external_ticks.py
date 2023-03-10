# For details on your rights to use the actual data downloaded. Remember - the Yahoo! finance API is intended for personal use only.

import os
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
import argparse

yf.pdr_override() # <== that's all it takes :-)


DATA_DIR = os.path.join(os.path.curdir, 'yahoo_data')


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
    dataframe = pdr.get_data_yahoo(name, start=start, end=end)
    return dataframe


def build_args():
    parser = argparse.ArgumentParser()
    today = datetime.date.today()
    parser.add_argument('--start', help='the start date of the stock', type=str, default="2017-01-01")
    parser.add_argument('--end', help='the end date of the stock', type=str, default=today.strftime("%Y-%m-%d"))
    parser.add_argument('--name', help='the name of the stock', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = build_args()
    print('fetching ' + args.name)
    df = fetch_data(args.name, args.start, args.end)
    df.to_csv(os.path.join(DATA_DIR, '{}.csv'.format(args.name)))


