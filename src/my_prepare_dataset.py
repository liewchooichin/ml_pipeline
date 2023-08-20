# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:28:06 2023.

@author: Kang Liew Bei

Prepare the dataframe into numpy for learning.
"""

from my_std_lib import RND_NUM
from my_std_lib import output_path
from my_get_db import write_df_to_sql
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd


def make_numpy(dataframe, feature_names, label_name):
    """
    Prepare dataframe into numpy features and label.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with features and a label.
    feature_names : list of str
        List of feature names.
    label_name : str
        Name of the label.

    Returns
    -------
    X : ndarray of shape (nsamples, nfeatures)
        Features X.
    y : ndarray of shape (nsamples, )
        Label y.

    """
    # get the features and label dataframe
    df_features = dataframe[feature_names]
    ser_label = dataframe[label_name]

    # convert the features and label into numpy
    X = df_features.to_numpy()
    y = ser_label.to_numpy()
    print(f"{X.shape=} features")
    print(f"{y.shape=} label")

    return X, y


def make_train_test(dataframe, feature_names,
                    label_name, table_name, test_size=0.2):
    """
    Make a dataframe into a numpy array and split into train, test set.

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
    df_X = dataframe[feature_names]
    df_y = dataframe[label_name]

    # split the original data into
    # test and train set
    df_X_train, df_X_test, df_y_train, df_y_test = \
        train_test_split(
            df_X, df_y,
            test_size=test_size,
            shuffle=True,
            stratify=df_y,
            random_state=RND_NUM
        )

    # Save a test dataframe in pd.DataFrame form.
    filename = output_path + table_name + '_df_X_test.pkl'
    j = joblib.dump(df_X_test, filename)
    print(f"{j} pickled X_test dataframe")
    filename = output_path + table_name + '_df_y_test.pkl'
    j = joblib.dump(df_y_test, filename)
    print(f"{j} pickled y_test dataframe")
    # concatenate this dataframe and write to sqlite3 db
    df_y_test = pd.DataFrame(df_y_test, columns=[label_name])
    df_test = pd.concat([df_X_test, df_y_test], axis=1)
    # save this test dataframe to a sqlite3 db also
    # table name is appended with '_test'
    write_df_to_sql(df_test, 'score_test')

    # Get the numpy array
    # X, y = make_numpy(dataframe, feature_names, label_name)
    # convert the features and label into numpy
    X_train = df_X_train.to_numpy()
    X_test = df_X_test.to_numpy()
    y_train = df_y_train.to_numpy()
    y_test = df_y_test.to_numpy()
    print(f"{X_train.shape=}")
    print(f"{X_test.shape=}")
    print(f"{y_train.shape=}")
    print(f"{y_test.shape=}")

    # save the train-test for later use,
    # so that the splitting step is not necessary for
    # every testing.
    # save in ndarray form.
    save_train_test(table_name, X_train, X_test, y_train, y_test)


def save_train_test(table_name, X_train, X_test, y_train, y_test):
    """
    Save the train-test set for later use.

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
    X : ndarray of shape (nsamples, nfeatures)
        Features X.
    y : ndarray of shape (nsamples, )
        Label y.

    """
    # Save the train-test set
    filename = output_path + table_name + '_X_train.pkl'
    j = joblib.dump(X_train, filename)
    print(f"{j} pickled {filename}")

    filename = output_path + table_name + '_X_test.pkl'
    j = joblib.dump(X_test, filename)
    print(f"{j} pickled {filename}")

    filename = output_path + table_name + '_y_train.pkl'
    j = joblib.dump(y_train, filename)
    print(f"{j} pickled {filename}")

    filename = output_path + table_name + '_y_test.pkl'
    j = joblib.dump(y_test, filename)
    print(f"{j} pickled {filename}")


def load_Xy_set(table_name, kind='train'):
    """
    Load the numpy X and y have been save using table_name.

    Parameters
    ----------
    table_name : str
        Name of the table.
    kind : either 'train' or 'test'
        Load train or test set. The default is 'train'.

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    """
    X = None
    y = None
    if kind == 'train':
        # get a train set
        filename = output_path + table_name + '_X_train.pkl'
        print(f"Loading {filename}")
        X = joblib.load(filename)
        print(f"X_train loaded. {X.shape=}")

        filename = output_path + table_name + '_y_train.pkl'
        print(f"Loading {filename}")
        y = joblib.load(filename)
        print(f"y_train loaded. {y.shape=}")
    elif kind == 'test':
        # get a test set
        filename = output_path + table_name + '_X_test.pkl'
        print(f"Loading {filename}")
        X = joblib.load(filename)
        print(f"X_test loaded. {X.shape=}")

        filename = output_path + table_name + '_y_test.pkl'
        print(f"Loading {filename}")
        y = joblib.load(filename)
        print(f"y_test loaded. {y.shape=}")

    return X, y
