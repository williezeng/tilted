ETHEREUM_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', 'ETH-USD')]
STOCK_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', '{string_name}')]

YAHOO_DATA_DIR = 'yahoo_data'
STOCK_DATA_DIR = 'data'

BANNED_TICKERS = ['CEG', 'AMCR', 'HWM'] # These companies did not exist on the starting date, therefore their data is shorter


CONCATENATED_INDICATORS_FILE = '00_concatenated_indicators.csv'
CONCATENATED_BUY_SELL_SIGNALS_FILE = '00_concatenated_buy_sell_signals.csv'
PREDICTION_FILE = '00_predictions.csv'
SAVED_MODEL_FILE = '00_saved_model.joblib'
TRAINING_DATA_DIR = 'training_data'
TESTING_DATA_DIR = 'testing_data'
