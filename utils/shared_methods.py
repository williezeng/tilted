import os
import pandas as pd
import multiprocessing
import joblib

from utils import constants
from sklearn.metrics import accuracy_score


def get_absolute_file_paths(data_dir):
    file_paths = []
    for filename in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, filename)) and filename.endswith('.csv') and not filename.startswith('00_'):
            file_paths.append(os.path.join(data_dir, filename))
    return file_paths


def get_technical_indicators_and_buy_sell_dfs(file_path):
    technical_indicators_df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])
    buy_sell_signal_df = technical_indicators_df.pop('bs_signal')
    return technical_indicators_df, buy_sell_signal_df, file_path


def parallel_get_technical_indicators_and_buy_sell_dfs(list_of_data_files):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(get_technical_indicators_and_buy_sell_dfs, list_of_data_files)
    return results


def write_prediction_to_csv(predictions_and_file_path):
    prediction_df, prediction_file_path = predictions_and_file_path
    prediction_df.to_csv(prediction_file_path)


def save_predictions_and_accuracy():
    # get all test data csv data frames
    # sort based on ticker name
    # predict and calculate accuracy
    # save model
    # save accuracy

    results = parallel_get_technical_indicators_and_buy_sell_dfs(get_absolute_file_paths(constants.TESTING_DATA_DIR_PATH))
    accuracy_list = []
    report = []
    predictions_and_file_path = []
    sorted_results = sorted(results, key=lambda x: x[2])
    print('Predicting and calculating accuracy')
    for result in sorted_results:
        model = joblib.load(constants.SAVED_MODEL_FILE_PATH)
        reference_technical_indicator_df, reference_buy_sell_df, test_data_file_path = result
        file_name = test_data_file_path.split("/")[-1]
        model_predictions = pd.DataFrame(model.predict(reference_technical_indicator_df),
                                         index=reference_technical_indicator_df.index, columns=['bs_signal'])
        predictions_and_file_path.append((model_predictions, os.path.join(constants.TESTING_PREDICTION_DATA_DIR_PATH, file_name)))
        accuracy = accuracy_score(reference_buy_sell_df, model_predictions)
        accuracy_list.append(accuracy)
        report.append(f"{accuracy} for {file_name}")
    print('Saving model')

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        __ = pool.map(write_prediction_to_csv, predictions_and_file_path)

    print('Saving Accuracy')
    total = sum(accuracy_list)
    average = total / len(accuracy_list) if accuracy_list else 0
    print("The average test accuracy is:", average)
    with open(constants.TESTING_PREDICTION_ACCURACY_FILE, 'w') as file:
        for accuracy_line in report:
            file.write(accuracy_line + '\n')
        file.write(f'average test score {average} \n')

    return accuracy_list, report
