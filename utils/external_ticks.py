# For details on your rights to use the actual data downloaded. Remember - the Yahoo! finance API is intended for personal use only.

import yfinance as yf
from dateutil import parser
import datetime
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr
import pandas
import yfinance as yf
import argparse

yf.pdr_override() # <== that's all it takes :-)


# 2017-11-09 first ethereum tick
# header = [('Close', 'ETH-USD'), ('High', 'ETH-USD'), ('Low', 'ETH-USD'), ('Open', 'ETH-USD'), ('Volume', 'ETH-USD')]
# ethereum_df = data.loc[:, header]
# ethereum_df.to_csv('ethereum_df.csv')
# import pdb
# pdb.set_trace()
# # pandas.DataFrame.to_csv('idk/')/
# today = datetime.datetime.now()
# a_year_ago = today - relativedelta(years=1)
# iso_date = today.isoformat()
#
# import pdb
# pdb.set_trace()
# data = yf.download("ETH-USD", start="2017-01-01", end="2017-04-30")


def fetch_data(name, start, end):
    assert isinstance(name, str), '%s must be a str'.format(names)
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


def write_individual_df_to_file(name, dataframe, header=None):
    assert isinstance(header, list), 'header must be a list of tuples'
    assert isinstance(header[0], tuple), 'header must be a list of tuples'
    if header:
        individual_df = dataframe.loc[:, header]
    else:
        individual_df = dataframe
    individual_df.to_csv('%s.csv'.format(name))


def read_file(name):
    df = pandas.read_csv(name, index_col=[0], header=[0, 1], skipinitialspace=True)
    return df

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', help='the start date of the stock', type=str, default="2017-01-01")
    parser.add_argument('--end', help='the end date of the stock', type=str, default="2022-01-01")
    parser.add_argument('--name', help='the name of the stock', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = build_args()
    print('fetching ' + args.name)
    df = fetch_data(args.name, args.start, args.end)
    df.to_csv('{}.csv'.format(args.name))


