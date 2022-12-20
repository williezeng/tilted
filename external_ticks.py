# For details on your rights to use the actual data downloaded. Remember - the Yahoo! finance API is intended for personal use only.

import yfinance as yf
from dateutil import parser
import datetime
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr
import pandas
import yfinance as yf
import constants

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


def fetch_data(names, start, end):
    assert isinstance(names, list), '%s must be a list'.format(names)
    assert isinstance(start, str), '%s must be a str'.format(start)
    assert isinstance(end, str), '%s must be a str'.format(end)

    def validate(date_text):
        try:
            datetime.datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
    validate(start)
    validate(end)
    dataframe = pdr.get_data_yahoo(' '.join(names), start=start, end=end)
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
    df2 = pandas.read_csv(name, index_col=[0], header=[0, 1], skipinitialspace=True)
    return df2




if __name__ == "__main__":
    #build arg parse
    # arg parse
    # call fetch_data

    # write_individual_df_to_file('%s_df'.format(filename), dataframe, header=constants.ETHEREUM_HEADER)
    read_file('real_eth.csv')

