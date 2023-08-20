# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:31:55 2023.

@author:Liew

Import frequently used variables.
"""

import os

# mini test to test small size data to
# make sure everything runs well.
mini_test = False

# name of the table or dataset name.
# use short name.
table_name = 'score'

# global variables
RND_NUM = 10

# scorer
multi_scoring = ['neg_mean_squared_error',
                 'neg_root_mean_squared_error',
                 'neg_mean_absolute_error',
                 'neg_median_absolute_error'
                 ]
scoring = 'neg_root_mean_squared_error'

# various files and paths
data_path = '../data/'
output_path = '../output/'
output_file = output_path + 'output.txt'  # output info
feature_conf = output_path + 'features.conf'  # features used
models_conf = output_path + 'models.conf'  # model config file
cv_score_conf = output_path + 'cv_score.conf'  # cv score
pred_conf = output_path + 'prediction_score.conf'  # prediction scores
importance_conf = output_path + 'importance.conf'  # feature importances

# params of model which can be changed
params_conf = output_path + 'params.conf'


def check_file_directory():
    """
    Check if the data and output directory exist.

    Returns
    -------
    None.

    """
    # Check if the file directory exists.
    # If the directory does not exist, make
    # the directory.
    if os.path.exists(data_path) is False:
        os.mkdir(data_path)
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
