# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:28:06 2023

@author: Kang Liew Bei

Prepare the dataframe into numpy for learning.
"""

from my_std_lib import *
from sklearn.model_selection import train_test_split


def make_train_test(dataframe, feature_names, label_name, table_name, test_size=0.2):
    """
    Make a dataframe without any null values into a numpy
    array and split into train, test set.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe of the dataset with preprocessed features.
    feature_names : list of str
        All the feature names to be included in the learning.
    label_name : str
        The label of the dataset.

    Returns
    -------
    None.
    X_train, X_test, y_train, y_test are save in pickle. Ready to be
    fed into learning algorithms.

    """

    # get the features and label dataframe
    df_features = dataframe[feature_names]
    ser_label = dataframe[label_name]

    # convert the features and label into numpy
    X = df_features.to_numpy()
    y = ser_label.to_numpy()
    print(f"{X.shape=} features")
    print(f"{y.shape=} label")

    # split the original data into
    # test and train set
    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y,
            test_size=test_size,
            shuffle=True,
            stratify=y,
            random_state=RND_NUM
        )

    # save the train-test for later use,
    # so that the splitting step is not necessary for
    # every testing
    save_train_test(table_name, X_train, X_test, y_train, y_test)


def save_train_test(table_name, X_train, X_test, y_train, y_test):
    """
    Save the train-test set for later use. The train_test_split
    is not called for every testing.

    Parameters
    ----------
    table_name : str
        Name of the table for saving the pickle file.
    X_train : ndarray of shape (nsamples, nfeatures)
        Features for training.
    X_test : ndarray of shape (nsamples, nfeatures)
        Features for prediction.
    y_train : ndarray of shape (nfeatures, )
        Label for training.
    y_test : ndarray of shape (nfeatures, )
        Label for prediction.

    Returns
    -------
    None.

    """
    # Save the train-test set
    filename = data_path + table_name + '_X_train.pkl'
    j = joblib.dump(X_train, filename)
    print(j)

    filename = data_path + table_name + '_X_test.pkl'
    j = joblib.dump(X_test, filename)
    print(j)

    filename = data_path + table_name + '_y_train.pkl'
    j = joblib.dump(y_train, filename)
    print(j)

    filename = data_path + table_name + '_y_test.pkl'
    j = joblib.dump(y_test, filename)
    print(j)
