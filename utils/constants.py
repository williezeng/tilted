ETHEREUM_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', 'ETH-USD')]
STOCK_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', '{string_name}')]

YAHOO_DATA_DIR = 'yahoo_data'
STOCK_DATA_DIR = 'data'

BANNED_TICKERS = ['CEG', 'AMCR', 'HWM'] # These companies did not exist on the starting date, therefore their data is shorter