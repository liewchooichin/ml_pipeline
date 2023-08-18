# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:31:55 2023.

@author:Liew

Import frequently used variables.
"""

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
pred_conf = output_path + 'pred.conf'  # prediction scores
importance_conf = output_path + 'importance.conf'  # feature importances

# params of model which can be changed
params_conf = output_path + 'params.conf'
