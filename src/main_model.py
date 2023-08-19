# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:27:17 2023

@author: Kang Liew Bei

Main module to train models. It will
load the X and y from pickles prepared
during the preprocessing steps.
"""

from my_std_lib import mini_test
from my_std_lib import table_name
from my_std_lib import output_file
from my_std_lib import models_conf
from my_std_lib import cv_score_conf
from my_std_lib import output_path
from my_model_maker import get_models_available
from my_model_maker import make_model
from my_prepare_dataset import load_Xy_set
from my_std_lib import check_file_directory
from time import time


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


if __name__ == "__main__":
    # check if ouput and data directory exists
    check_file_directory()

    # get a train set
    X_train, y_train = load_Xy_set(table_name, kind='train')
    print("Train set loaded \n"
          f"{X_train.shape=} \n"
          f"{y_train.shape=}"
          )

    # make all the models
    # run mini test to make sure everything is find
    print(f"This is a mini test: {mini_test}")
    if mini_test is True:
        make_all_models(X_train[:50, :], y_train[:50])
    elif mini_test is False:
        make_all_models(X_train, y_train)

    print("--- End of program ---")
