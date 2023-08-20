# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:59:35 2023.

@author: Kang Liew Bei

Making predictions.
Prediction scores is in output folder.
"""

from my_std_lib import table_name
from my_std_lib import output_file
from my_std_lib import pred_conf
from my_model_eval import get_prediction_score
from my_model_eval import load_model
from my_model_maker import get_models_available
from my_prepare_dataset import load_Xy_set
from my_std_lib import check_file_directory
from configparser import ConfigParser


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
        # each estimator has one section in the
        # config file
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
    with open(pred_conf, 'w') as configfile:
        config.write(configfile)

    print("Finish prediction")


# Main
if __name__ == "__main__":

    # check if ouput and data directory exists
    check_file_directory()

    # The ready made numpy X, y set has been
    # saved to pickle file.
    # Now, this is ready to be loaded to
    # prediction.
    # Load the test set.
    X_test, y_test = load_Xy_set(table_name, kind='test')
    print("Test set loaded \n"
          f"{X_test.shape=} \n"
          f"{y_test.shape=}"
          )

    # Get predictions of all the models.
    # Result of scores are stored in output dir.
    # Fitted models have been saved to pickle files.
    # A quick funtion to make predictions for all models.
    print("Making predictions ...")
    make_all_prediction(X_test, y_test)

    print("--- End of program ---")
