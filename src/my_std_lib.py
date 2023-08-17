# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:31:55 2023.

@author:Liew

Import frequently used libraries.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

from sklearn.model_selection import cross_validate


from sklearn.datasets import load_diabetes

import joblib
from time import time
import os

from configparser import ConfigParser

# global variables
RND_NUM = 10

# scorer
multi_scoring = ['neg_mean_squared_error',
                 'neg_root_mean_squared_error',
                 'neg_mean_absolute_error',
                 'neg_median_absolute_error'
                 ]
scoring = 'neg_root_mean_squared_error'


data_path = '../data/'
output_path = '../output/'
output_file = '../output/output.txt'  # output info
models_conf = '../output/models.conf'  # model config file
cv_score_conf = '../output/cv_score.conf'  # cv score

