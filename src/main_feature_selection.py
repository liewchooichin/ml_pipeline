# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:16:59 2023.

@author: Kang Liew Bei

Perform feature selections.
And get the dataframe into ndarray form.
Features can be turned on or off to
test the performance of combination of
different features.

This step is not necessary if no features
selection is needed.
"""

from my_std_lib import table_name
from my_std_lib import feature_conf
from my_preprocessing import prep_load
from my_prepare_dataset import make_train_test
from configparser import ConfigParser
import os


def read_features_from_file():
    """
    Read features and return the name of
    features and label. The feature can
    be turned on or off.

    Raises
    ------
    FileNotFoundError
        DESCRIPTION.

    Returns
    -------
    feature_names : array of str
        Name of features selected.
    label_name : strE
        Name of label.

    """
    # select features from the feature.conf
    # check for the features config file
    if os.path.exists(feature_conf) is False:
        raise FileNotFoundError(f'{feature_conf} is not found.')
    config = ConfigParser()
    config.read(feature_conf)
    label_name = config['label']['name']
    print(f"{label_name} - label name")
    # From the features_conf, selected features is True,
    # if false, the features is not selected for training.
    feature_names = []
    for k, v in config['features'].items():
        if v == 'True':
            feature_names.append(k)
    print(f"{len(feature_names)} selected features")
    for i in feature_names:
        print(f"- {i}")

    return feature_names, label_name


def select_features():
    """
    Select features to test their effects.
    Read from the prepared dataframe pickle.
    Get the selected features and convert
    into train-test set.

    Raises
    ------
    FileNotFoundError
        The features file is not found.

    Returns
    -------
    None.

    """
    # read features from the default conf file
    # get the feature names and label name
    feature_names, label_name = read_features_from_file()

    # load the preprocessed dataframe
    df_prep = prep_load()

    # make the train test set
    # X_train, X_test, y_train, y_test are in
    # pickle files.
    make_train_test(df_prep, feature_names, label_name, table_name)


# Main
if __name__ == "__main__":

    # select features to test their effects
    # on the evaluation metrics.
    # The train-test set will also be
    # saved for later use.
    print("Select features ...")
    select_features()

    print("--- End of program ---")
