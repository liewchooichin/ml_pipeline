# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:25:10 2023

@author: Kang Liew Bei
"""

from my_std_lib import *
from model_maker import get_models_available
from model_maker import make_model
from my_preprocessing import process_all_cols
from prepare_dataset import make_train_test


def make_all_models(X_train, y_train):
    """
    Make all the models.

    Parameters
    ----------
    X_train : ndarray of shape (nsamples, nfeatures)
        Features for training.
    y_train : ndarray of shape (nsamples, )
        Label for training.

    Returns
    -------
    None. Result are written to 'output.txt'. Parameters of models
    are written to 'models.conf'. CV scores are written to 
    'cv_score.conf' 

    """
    # Model to use
    model_list = get_models_available(kind='regressor')

    # Display where the information is
    for estimator_name in model_list:
        print(f"training {estimator_name} ...")

        start_time = time()
        make_model(estimator_name, X_train, y_train)

        elapsed_time = time() - start_time
        print(f"{elapsed_time/60} min taken")

        model_file = output_path + estimator_name + ".pkl"
        print(f"{model_file} - model ready for use")

    print(f"{output_file} - outputs for easy reading")
    print(f"{cv_score_conf} - cv scores")
    print(f"{models_conf} - parameters of models")


##########
## Main ##
##########
if __name__ == '__main__':

    # check if the file directory exists
    if os.path.exists(data_path) == False:
        os.mkdir(data_path)
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)

    table_name = 'score'
    dataframe_file = data_path + 'score.pkl'

    try:
        raw = joblib.load(dataframe_file)
    except FileNotFoundError():
        print("File not found")

    # process all features
    df_prep = process_all_cols(raw)

    # collect the feature names
    feature_names = [s for s in df_prep.columns]
    label_name = 'final_test'
    feature_names.remove(label_name)

    # make the train test set
    # X_train, X_test, y_train, y_test are in
    # pickle files.
    make_train_test(df_prep, feature_names, label_name, table_name)

    # get a train set
    filename = data_path + table_name + '_X_train.pkl'
    print(f"Loading {filename}")
    X_train = joblib.load(filename)
    print(f"X_train loaded. {X_train.shape=}")

    filename = data_path + table_name + '_y_train.pkl'
    print(f"Loading {filename}")
    y_train = joblib.load(filename)
    print(f"y_train loaded. {y_train.shape=}")

    # mini test to make sure everything runs fine
    make_all_models(X_train[:50, :], y_train[:50])
    # the real train set
    # make_all_models(X_train, y_train)
