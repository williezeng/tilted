import os

ETHEREUM_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', 'ETH-USD')]
STOCK_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', '{string_name}')]
BANNED_TICKERS = ['SW', 'CEG', 'AMCR', 'HWM']  # These companies did not exist on the starting date, therefore their data is shorter

TECHNICAL_INDICATORS = ['VWMA_10', 'VWAP', 'SMA_10', 'EMA_10', 'bb_upper', 'bb_lower', 'bb_width', 'RSI', 'AMO', 'Close']


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
TESTING_PREDICTION_DATA_DIR_PATH = os.path.join(TESTING_PREDICTION_DIR_PATH, 'data')
TESTING_PREDICTION_GRAPHS_DIR_PATH = os.path.join(TESTING_PREDICTION_DIR_PATH, 'graphs')

PARENT_REPORT_DIRECTORY_NAME = 'reports'
FULL_REPORT_DIR = os.path.join(PARENT_REPORT_DIRECTORY_NAME, '{}_{}')
BACKTESTING_RESULT_FILE_NAME = 'backtesting_results.txt'
SUMMARY_REPORT_FILE_NAME = "summary_report.txt"
GENERATED_REPORTS = [BACKTESTING_RESULT_FILE_NAME, SUMMARY_REPORT_FILE_NAME]

MULTIPROCESS_CPU_NUMBER = 4

SHUFFLE_RANDOM_STATE = 63

RANDOM_FOREST_RANDOM_STATE = 63
RANDOM_FOREST_CLASS_WEIGHT = {0: 1.1, -1: 1.1, 1: 1}

# Threshold needs to be higher than trade impact + commission
# threshold to days = 0.01:2
BUY_THRESHOLD = 0.005
SELL_THRESHOLD = -BUY_THRESHOLD
LOOK_AHEAD_DAYS_TO_GENERATE_BUY_SELL = 1

LONG_TERM_PERIOD = 10
LOOK_BACK_PERIOD = 5
BUY = 1
SELL = -1
HOLD = 0

INITIAL_CAP = 10000
