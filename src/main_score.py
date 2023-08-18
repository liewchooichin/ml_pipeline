# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:25:10 2023.

@author: Kang Liew Bei

The main file. Use for preprocessing of
raw data, training models and prediction.

Use mini_test=True or False to toggle whether
to do a fast test to make sure everything runs
well.
mini_test=False to do the long training.

This is intended for my own testing purpose. The user
program is in another module.
"""


from my_std_lib import output_file
from my_std_lib import feature_conf
from my_std_lib import cv_score_conf
from my_std_lib import models_conf
from my_std_lib import pred_conf
from my_std_lib import importance_conf
from my_std_lib import output_path
from my_std_lib import data_path

from my_model_maker import get_models_available
from my_model_maker import make_model
from my_model_eval import get_prediction_score
from my_model_eval import load_model
from my_model_eval import get_importance
from my_preprocessing import prep_save
from my_preprocessing import prep_load
from my_prepare_dataset import make_train_test
from my_prepare_dataset import load_Xy_set
from configparser import ConfigParser
from time import time
import os
import joblib

# Do a mini-test or real test
# Fast mini-test to make sure everything
# runs well.
mini_test = True


def get_important_features(X_train, y_train):
    """
    Inspect the importance of each features is for an estimator.

    Parameters
    ----------
    X_test : ndarray of shape(nsamples, nfeatures)
        Features from training set can be used.
    y_test : TYPE
        Label from training set can be used.

    Returns
    -------
    None.

    """
    # Store score to output file
    f = open(output_file, "a")

    # Also write the score to a config file
    config = ConfigParser()

    # Model to use
    model_list = get_models_available(kind='regressor')

    # Display where the information is
    for estimator_name in model_list:
        print(f"feature importances {estimator_name} ...")

        # each estimator has one section
        config[estimator_name] = {}

        # load model
        model = load_model(estimator_name)
        print(f"{estimator_name} - loaded")

        start_time = time()
        impt = get_importance(model, X_train, y_train)

        elapsed_time = time() - start_time
        print(f"{elapsed_time/60} min taken")

        for k, v in impt.items():
            config[estimator_name][k] = str(v)
            f.write(f"{k}: {v} \n")
            # print(f"{k}: {v}") # too verbose

    # close the file
    f.close()

    # write and close the config
    with open(importance_conf, 'a') as configfile:
        config.write(configfile)

    print("Finish feature importances")


def make_all_prediction(X_test, y_test):
    """
    Make prediction from the models built.

    Parameters
    ----------
    X_test : ndarray of shape(nsamples, nfeatures)
        Features for testing.
    y_test : ndarray of shape(nsamples, )
        Label for testing.

    Returns
    -------
    Multiple scoring parameters defined in my standard library.

    """
    # Store score to output file
    f = open(output_file, "a")
    # Also write the score to a config file
    config = ConfigParser()

    # Model to use
    model_list = get_models_available(kind='regressor')

    # Display where the information is
    for estimator_name in model_list:
        print(f"prediction {estimator_name} ...")
        # each estimator has one section
        config[estimator_name] = {}
        # load model
        model = load_model(estimator_name)
        f.write(f"{estimator_name} \n")
        print(f"{estimator_name} - loaded")

        # mini test to make prediction to make sure everything
        # runs fine
        score = get_prediction_score(model, X_test, y_test)
        for k, v in score.items():
            config[estimator_name][k] = str(v)
            f.write(f"{k}: {v} \n")
            print(f"{k}: {v}")

    # close the file
    f.close()

    # write and close the config
    with open(pred_conf, 'a') as configfile:
        config.write(configfile)

    print("Finish prediction")


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


# Main
if __name__ == '__main__':

    # check if the file directory exists
    if os.path.exists(data_path) is False:
        os.mkdir(data_path)
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # sqlite table name
    table_name = 'score'
    dataframe_file = data_path + table_name + '.pkl'

    try:
        raw = joblib.load(dataframe_file)
    except FileNotFoundError as e:
        print(f"File not found: {e}")

    # Process all features, save it to pickle,
    # then load the dataframe.
    # So that the prep dataframe can be loaded
    # for later use.
    # There is no need to go through the
    # preprocessing step everytime.
    prep_save(raw)
    df_prep = prep_load()

    # collect the feature names
    feature_names = [s for s in df_prep.columns]
    label_name = 'final_test'
    feature_names.remove(label_name)

    # write the features to a file.
    # A feature can be turned on or off.
    config = ConfigParser()
    config['features'] = {}
    for i in feature_names:
        config['features'][i] = 'True'
    config['label'] = {}
    config['label']['name'] = label_name
    with open(feature_conf, "w") as configfile:
        config.write(configfile)

    # make the train test set
    # X_train, X_test, y_train, y_test are in
    # pickle files.
    make_train_test(df_prep, feature_names, label_name, table_name)

    # get a train set
    X_train, y_train = load_Xy_set(table_name, kind='train')

    # load the test set
    X_test, y_test = load_Xy_set(table_name, kind='test')

    # Doing mini-test or real test
    if mini_test is True:
        # mini test to make sure everything runs fine
        make_all_models(X_train[:50, :], y_train[:50])
        # get_important_features(X_train[:10, :], y_train[:10])
        make_all_prediction(X_test[:5], y_test[:5])
    elif mini_test is False:
        # the real train set
        make_all_models(X_train, y_train)
        # get_important_features(X_train, y_train)
        make_all_prediction(X_test, y_test)
    else:
        print("Not doing any training or testing.")

    # end
    print("--- End of program ---")
