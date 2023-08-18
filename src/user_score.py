# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:38:39 2023

@author: Kang Liew Bei

For user program.

The db file must be in path '../data/'.
The filename must be in the form 'table_name.db'.
For example, sqlite table name is 'score'.
The file must be '../data/score.db'.
The features can be selected by editing output/features.conf.
Set the features by True or False.
"""

from my_std_lib import RND_NUM
from my_std_lib import data_path
from my_std_lib import output_path
from my_std_lib import feature_conf
from my_std_lib import params_conf
from my_get_db import read_sqlite3_from_file
from my_preprocessing import process_all_cols
from my_prepare_dataset import make_numpy
from main_score import make_all_models
from main_score import make_all_prediction
from my_model_maker import get_models_available
from my_model_maker import make_model
from my_model_eval import load_model
from my_model_eval import get_prediction_score
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from configparser import ConfigParser
from time import time
import os
import joblib
import sys

# table name of the db
table_name = 'score'

# mini test to test small size data to
# make sure everything runs well.
mini_test = True


def _get_params_conf(estimator_name):
    # read from the params conf
    config = ConfigParser()
    config.read(params_conf)
    # get the parameters of the estimator
    # given.
    # need to add the '__' for pipeline
    # parameters.
    # As for changing the types, is there any
    # better ways to do this?
    # Why the config writer writes everything in
    # lowercase?
    print(f"{estimator_name} getting parameters ...")
    params = dict()
    for k, v in config[estimator_name].items():
        # special case of C - somehow it is always
        # lowercase in the config writer.
        if k == 'c':
            k = 'C'
        param_name = estimator_name + '__' + k
        # need to type cast into proper type
        # test for bool, float and int
        if v == 'None':
            proper_value = None
        elif (v == 'True') or (v == 'False'):
            proper_value = bool(v)
        elif ('.' in v) and (v[0].isnumeric()):
            proper_value = float(v)
        elif ('.' not in v) and (v[0].isnumeric()):
            proper_value = int(v)
        elif v[0] == '-':
            proper_value = int(v)
        else:
            proper_value = v
        print(f"{param_name} = {proper_value}, {type(proper_value)}")

        # need to make the value a [] because of
        # the param_grid.
        params[param_name] = [proper_value]

    return params


def _preprocess_select_features(df_test):
    # preprocess the raw dataframe
    df_prep = process_all_cols(df_test)

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

    # return the preprocessed dataframe, selected features
    # and label.
    return df_prep, feature_names, label_name


# Main
if __name__ == "__main__":
    # check if the file directory exists
    if os.path.exists(data_path) is False:
        os.mkdir(data_path)
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # Argument parsing
    parser = parser = ArgumentParser(
        prog='user_score',
        description='Evaluate the prediction scorings. '
        'Table_name must be in the path data/table_name.db. '
        'Provide only the table_name, e.g. score. '
        'Select features from output/features.conf. '
        'Set features to be selected by True or False. '
        'Model parameters can be changed. The output/params.conf '
        'contains the default parameters which can be changed. ',
        epilog='usage: user_score -m polynomial -f score -p no'
    )
    parser.add_argument(
        '-l', '--list', action='store_true',
        help='list the models available',
    )
    parser.add_argument(
        '-m', '--model', metavar='model', nargs='?', type=str,
        help='provide a model name, default sgd',
        default='polynomial'
    )
    parser.add_argument(
        '-f', '--file', metavar='table_name', nargs='?', type=str,
        help='provide the table name, default score',
        default='score'
    )
    parser.add_argument(
        '-p', '--params', metavar='params', nargs='?', type=str,
        help='model parameters yes or no, default no',
        default='no'
    )

    # get the argss
    args = parser.parse_args()

    # if there are no args, exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.list is True:
        print("List of models")
        print("all: make all models listed  here")
        print("polynomial: polynomial linear regression")
        print("svr: vector support machine")
        print("forest: random forest")
        print("sgd: stochastic gradient descent")
        print("knn: k nearest neighbors")
        print("output/features.conf: select features by True or False")
        print("output/params.conf: set parameters for estimator")
        sys.exit(0)

    # check if the db filename exists
    local_filename = data_path + args.file + '.db'
    print(f"{local_filename} - db to be read in")
    if os.path.exists(local_filename) is False:
        print(f"{local_filename} is not found.")
        sys.exit(0)

    # check if model name is valid
    model_list = get_models_available()
    # there is one more 'all' models option
    model_list.append('all')
    if args.model not in model_list:
        print("Model not found. "
              "Use -l or --list to see the list of models.")
        sys.exit(0)

    # if parameters is specified, check if the
    # output/params.conf exists
    params = dict()  # default to none
    if args.params == 'yes':
        if os.path.exists(params_conf) is False:
            print(f"{params_conf} is not found.")
            sys.exit(0)
        else:
            params = _get_params_conf(args.model)
    elif args.params == 'no':
        params = None
    else:
        print("-p or --params only takes yes or no")
        sys.exit(0)

    # read the db
    # the table name is in args.file
    df_test = read_sqlite3_from_file(args.file, local_filename)
    # preprocess the raw dataframe and select features
    # based on features.conf
    df_prep, feature_names, label_name = _preprocess_select_features(df_test)

    # make the numpy array
    X_prep, y_prep = make_numpy(df_prep, feature_names, label_name)
    # split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep,
        test_size=0.2,
        shuffle=True,
        stratify=y_prep,
        random_state=RND_NUM
    )

    # mini test for testing
    if mini_test is True:
        # mini test to make sure everything runs fine
        make_model(args.model, X_train[:50, :], y_train[:50], params)
        model = load_model(args.model)
        score = get_prediction_score(model, X_test[:5, :], y_test[:5])
    elif mini_test is False:
        # the real train set
        print(" Training in progress ...")
        start_time = time()
        make_model(args.model, X_train, y_train)
        model = load_model(args.model)
        score = get_prediction_score(model, X_test, y_test)
        # write the score
        for k, v in score.items():
            print(f"{k}: {v}")
        elapsed_time = time() - start_time
        print(f"{elapsed_time/60} min - time taken")
    else:
        print("Not doing any training or testing.")

    # end
    print("--- End of program ---")
