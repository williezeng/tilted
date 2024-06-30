import os

ETHEREUM_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', 'ETH-USD')]
STOCK_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', '{string_name}')]
BANNED_TICKERS = ['CEG', 'AMCR', 'HWM'] # These companies did not exist on the starting date, therefore their data is shorter

YAHOO_DATA_DIR = 'yahoo_data'
STOCK_DATA_DIR = 'data'

TRAINING_BASE_DIRECTORY_NAME = "training"
TESTING_BASE_DIRECTORY_NAME = 'testing'

TRAINING_COMBINED_DATA_DIR_PATH = os.path.join(TRAINING_BASE_DIRECTORY_NAME, 'combined_data')
TRAINING_DATA_DIR_PATH = os.path.join(TRAINING_BASE_DIRECTORY_NAME, "data")
TRAINING_GRAPHS_DIR_PATH = os.path.join(TRAINING_BASE_DIRECTORY_NAME, "graphs")

TESTING_DATA_DIR_PATH = os.path.join(TESTING_BASE_DIRECTORY_NAME, "data")
TESTING_GRAPHS_DIR_PATH = os.path.join(TESTING_BASE_DIRECTORY_NAME, "graphs")

SAVED_MODEL_FILE_PATH = os.path.join(TESTING_BASE_DIRECTORY_NAME, '00_saved_model.joblib')
PREDICTION_FILE = os.path.join(TESTING_BASE_DIRECTORY_NAME, '00_predictions.csv')

TRAINING_CONCATENATED_INDICATORS_FILE = os.path.join(TRAINING_COMBINED_DATA_DIR_PATH, '00_concatenated_indicators.csv')
TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE = os.path.join(TRAINING_COMBINED_DATA_DIR_PATH, '00_concatenated_buy_sell_signals.csv')

TESTING_PREDICTION_DIR_PATH = os.path.join(TESTING_BASE_DIRECTORY_NAME, 'predictions')
TESTING_PREDICTION_ACCURACY_DIR_PATH = os.path.join(TESTING_PREDICTION_DIR_PATH, 'accuracy')
TESTING_PREDICTION_DATA_DIR_PATH = os.path.join(TESTING_PREDICTION_DIR_PATH, 'data')
TESTING_PREDICTION_GRAPHS_DIR_PATH = os.path.join(TESTING_PREDICTION_DIR_PATH, 'graphs')
TESTING_PREDICTION_ACCURACY_FILE = os.path.join(TESTING_PREDICTION_ACCURACY_DIR_PATH, f'accuracy.txt')

SHUFFLE_RANDOM_STATE = 63

# Threshold needs to be higher than trade impact + commission
BUY_THRESHOLD = 0.05
SELL_THRESHOLD = -BUY_THRESHOLD
