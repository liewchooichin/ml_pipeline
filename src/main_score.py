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

It is the main place to preprocess the input
data. Then, the data will be train test split.
The prepared df in ready-preprocessed form in saved
in pickle so that it can be used by other modules
without going through the preprocessing steps again.
The prepared df is also split into the corresponding
train and test df_X and df_y forms.
The X and y ndarray for train and test will be
pickled for use by other modules.
"""

from my_std_lib import mini_test
from my_std_lib import table_name
from my_std_lib import data_path
from my_std_lib import feature_conf
from my_std_lib import check_file_directory
from my_preprocessing import prep_save
from my_preprocessing import prep_load
from my_prepare_dataset import make_train_test
# from main_feature_selection import select_features
from my_prepare_dataset import load_Xy_set
from main_prediction import make_all_prediction
from main_importances import get_all_important_features
from main_model import make_all_models
from configparser import ConfigParser
import joblib


# Main
if __name__ == '__main__':

    # check for output and data directories exist
    check_file_directory()

    # sqlite table name
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
    # A feature can be turned on or off
    # in another module. This is to write
    # the available features for experiment.
    config = ConfigParser()
    config['features'] = {}
    for i in feature_names:
        config['features'][i] = 'True'
    config['label'] = {}
    config['label']['name'] = label_name
    with open(feature_conf, "w") as configfile:
        config.write(configfile)

    # Make the train test set
    # X_train, X_test, y_train, y_test are in
    # pickle files.
    # for all features, no selection.
    # Either this line or the select_features must be written
    # to get the train-test set.
    make_train_test(df_prep, feature_names, label_name, table_name)
    # Select features and make them into train-test set.
    # Actually in this test file, the select_feature is not useful
    # because the features conf file will always get overwritten
    # by the code just before this.
    # This is just when run as individual pipeline.
    # This is put here as a testing to be turned on or off.
    # select_features()

    # get a train set
    X_train, y_train = load_Xy_set(table_name, kind='train')

    # load the test set
    X_test, y_test = load_Xy_set(table_name, kind='test')

    # Doing mini-test or real test
    print(f"This is a mini test: {mini_test}")
    if mini_test is True:
        # mini test to make sure everything runs fine
        make_all_models(X_train[:50, :], y_train[:50])
        get_all_important_features(X_train[:10, :], y_train[:10])
        make_all_prediction(X_test[:5], y_test[:5])
    elif mini_test is False:
        # the real train set
        # this step is a short hand for writing the necessary
        # models.conf file. The params.conf is a copy of this,
        # which can be used for setting parameters.
        make_all_models(X_train, y_train)
        # get_important_features(X_train, y_train)
        # make_all_prediction(X_test, y_test)
    else:
        print("Not doing any training or testing.")

    # end
    print("--- End of program ---")
