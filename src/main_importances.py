# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:30:47 2023

@author: Kang Liew Bei

Get the feature importances.
Main program to get feature importances.
Importances values are in output folder.
"""

from my_std_lib import table_name
from my_std_lib import output_file
from my_std_lib import importance_conf
from my_model_maker import get_models_available
from my_model_eval import get_importance
from my_model_eval import load_model
from my_prepare_dataset import load_Xy_set
from my_std_lib import check_file_directory
from configparser import ConfigParser
from time import time


def get_all_important_features(X_train, y_train):
    """
    Inspect the importance of each features for an estimator.

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
        impt = get_importance(X_train, y_train, model)

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


# Main
if __name__ == "__main__":
    # check if ouput and data directory exists
    check_file_directory()

    # Load the numpy X, y training set
    # that has been pickled.
    # The training set is used for
    # permutation of importances.
    X_train, y_train = load_Xy_set(table_name, kind='train')
    print("Train set loaded \n"
          f"{X_train.shape=} \n"
          f"{y_train.shape=}"
          )

    print("Permutation of feature importances ...")
    get_all_important_features(X_train, y_train)

    print("End of program")
