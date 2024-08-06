import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from catboost import CatBoostClassifier

ETHEREUM_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', 'ETH-USD')]
STOCK_HEADER = [('Close'), ('High'), ('Low'), ('Open'), ('Volume', '{string_name}')]
BANNED_TICKERS = ['SW', 'CEG', 'AMCR', 'HWM', 'GEV', 'SOLV']  # These companies have a very short time frame

TECHNICAL_INDICATORS = ['VWMA_10', 'VWAP', 'SMA_10', 'EMA_10', 'bb_upper', 'bb_lower', 'bb_width', 'RSI', 'AMO',
                        'Close']

default_file_type = '.parquet'

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

TRAINING_CONCATENATED_INDICATORS_FILE = os.path.join(TRAINING_COMBINED_DATA_DIR_PATH, '00_concatenated_indicators.parquet')
TRAINING_CONCATENATED_BUY_SELL_SIGNALS_FILE = os.path.join(TRAINING_COMBINED_DATA_DIR_PATH, '00_concatenated_buy_sell_signals.parquet')

TESTING_PREDICTION_DIR_PATH = os.path.join(TESTING_BASE_DIRECTORY_NAME, 'predictions')
TESTING_PREDICTION_DATA_DIR_PATH = os.path.join(TESTING_PREDICTION_DIR_PATH, 'data')
TESTING_PREDICTION_GRAPHS_DIR_PATH = os.path.join(TESTING_PREDICTION_DIR_PATH, 'graphs')

PARENT_REPORT_DIRECTORY_NAME = 'reports'
FULL_REPORT_DIR = os.path.join(PARENT_REPORT_DIRECTORY_NAME, '{}_{}')
SUMMARY_DIRECTORY = 'summary'
BACKTESTING_RESULT_FILE_NAME = '00_backtesting_results.txt'
SUMMARY_REPORT_FILE_NAME = "00_summary_report.txt"
GENERATED_REPORTS = [BACKTESTING_RESULT_FILE_NAME, SUMMARY_REPORT_FILE_NAME]

MULTIPROCESS_CPU_NUMBER = 4

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

SHUFFLE_RANDOM_STATE = 63

MODELS_RANDOM_STATE = 63
RANDOM_FOREST_CLASS_WEIGHT = {0: 1.1, -1: 1.3, 1: 1}

MODEL_MAP = {
    'random_forest': RandomForestClassifier,
    'catboost_cpu': CatBoostClassifier,
    'catboost_gpu': CatBoostClassifier,
    'adaboost': AdaBoostClassifier,
    'hist_gradient_boosting': HistGradientBoostingClassifier

}

base_estimator = DecisionTreeClassifier(criterion='gini',
                                        splitter='best',
                                        max_depth=1,
                                        min_samples_split=2,
                                        min_samples_leaf=20,
                                        min_weight_fraction_leaf=0.0,
                                        max_features=None,
                                        random_state=MODELS_RANDOM_STATE,
                                        max_leaf_nodes=None,
                                        min_impurity_decrease=0.0,
                                        class_weight=RANDOM_FOREST_CLASS_WEIGHT)

MODEL_ARGS = {
    'random_forest': {
        'n_estimators': 15,
        'max_depth': 25,
        'n_jobs': -1,
        'class_weight': RANDOM_FOREST_CLASS_WEIGHT,
        'random_state': MODELS_RANDOM_STATE
    },
    'catboost_cpu': {
        'iterations': 1000,
        'learning_rate': 0.33,
        'depth': 6,
        'task_type': 'CPU',
        'verbose': 0
    },
    'catboost_gpu': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'task_type': 'GPU',
        'devices': '0:1',
        'verbose': 0
    },
    'adaboost': {
        'estimator': base_estimator,
        'n_estimators': 200,
        'learning_rate': 1.0,
        'algorithm': 'SAMME.R',
        'n_jobs': '-1',
        'random_state': MODELS_RANDOM_STATE
    },
    'hist_gradient_boosting': {
        'learning_rate': 0.5,
        'max_iter': 300,
        'min_samples_leaf': 10,
        'max_depth': None,
        'max_leaf_nodes': 15,
        'warm_start': False,
        'random_state': MODELS_RANDOM_STATE,
        'verbose': 0,
        'tol': 0.0001
    }
}

