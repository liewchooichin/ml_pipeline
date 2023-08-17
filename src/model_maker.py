# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:47:12 2023

@author: Kang Liew Bei

Create models.
"""

from my_std_lib import *

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


def get_models_available(kind='all'):
    """
    Return a list of estimator available for use

    Parameters
    ----------
    kind : ['all', 'regressor', 'classifier']
        The kind of estimators requested. 
        Default return all estimators available.

    Returns
    -------
    A list of estimators available.

    """
    regressor = ['polynomial', 'sgd', 'svr', 'forest', 'knn']
    classifier = []
    if kind == 'all':
        return regressor + classifier
    elif kind == 'regressor':
        return regressor
    elif kind == 'classifier':
        return classifier
    else:
        return None


def make_model(estimator_name, X, y, param=None):
    """
    Make models available.

    Parameters
    ----------
    estimator_name : str
        Make the model and write all the parameters and scores.
    X : ndarray of shape (nsamples, nfeatures)
        Features for training.
    y : ndarray of shape (nsamples, )
    param : dict
        Parameters to be passed to GridSearchCV

    Returns
    -------
    None.

    """
    estimator_map = {
        'knn': make_knn_regressor,
        'polynomial': make_polynomial,
        'forest': make_forest_regressor,
        'svr': make_svr,
        'sgd': make_sgd_regressor
    }
    estimator_map[estimator_name](X, y, param)


def make_knn_regressor(X, y, param=None):
    # knn is based on distance. It will need scaled
    # data.
    knn = Pipeline(
        [
            ('std', StandardScaler()),
            ('knn', KNeighborsRegressor(algorithm='kd_tree'))
        ]
    )

    if param is not None:
        param_grid = param
    else:
        param_grid = {
            'knn__n_neighbors': [3, 5, 7],
            'knn__weights': ['uniform', 'distance'],
            'knn__p': [1, 2]
        }

    grid = do_grid_search(knn, param_grid, X, y)
    # save model, write params to config file, write ouput info
    estimator_name = 'knn'
    post_training(estimator_name, grid)


def make_forest_regressor(X, y, param=None):

    # limit the size of the tree
    forest = RandomForestRegressor(
        n_estimators=50,
        max_depth=30,
        min_samples_split=4,
        bootstrap=True, oob_score=False,
        random_state=RND_NUM, warm_start=True
    )

    if param is not None:
        param_grid = param
    else:
        # max_features: more randomness can be achieved by setting
        # smaller values, e.g. 0.3.
        param_grid = {
            'criterion': ['squared_error', 'absolute_error',
                          'friedman_mse', 'poisson'],
            'max_features': [0.3, 0.4, 0.5],
        }

    grid = do_grid_search(forest, param_grid, X, y)
    # save model, write params to config file, write ouput info
    estimator_name = "forest"
    post_training(estimator_name, grid)


def make_sgd_regressor(X, y, param=None):

    sgd_base = SGDRegressor(
        penalty='l2',
        random_state=RND_NUM,
        warm_start=True,
        early_stopping=True
    )
    sgd = Pipeline([
        ('std', StandardScaler()),
        ('sgd', sgd_base)
    ])

    if param is not None:
        param_grid = param
    else:
        # alpha increase by 5 times from the default values
        param_grid = {
            'sgd__loss': ['squared_error', 'huber',
                          'epsilon_insensitive',
                          'squared_epsilon_insensitive'],
            'sgd__alpha': [0.0001, 0.001, 0.005, 0.0125, 0.025]
        }

    grid = do_grid_search(sgd, param_grid, X, y)

    # save model, write params to config file, write ouput info
    estimator_name = 'sgd'
    post_training(estimator_name, grid)


def make_polynomial(X, y, param=None):

    linear = LinearRegression()
    poly_fea = PolynomialFeatures()

    poly = Pipeline([
        ('std', StandardScaler()),
        ('poly', poly_fea),
        ('linear', linear)
    ])

    if param is not None:
        param_grid = param
    else:
        param_grid = {
            'poly__degree': [1, 2, 3, 4, 5],
            'poly__interaction_only': [False, True],
            'poly__include_bias': [False, True]
        }

    grid = do_grid_search(poly, param_grid, X, y)

    # save model, write params to config file, write ouput info
    estimator_name = "polynomial"
    post_training(estimator_name, grid)


def make_svr(X, y, param=None):

    # kernel{‘poly’, ‘rbf’, ‘sigmoid’}
    # gamma{‘scale’, ‘auto’}
    # degree default 3
    if param is not None:
        param_grid = param
    else:
        param_grid = {
            'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svr__gamma': ['scale', 'auto'],
            'svr__degree': [1, 2, 3, 4, 5]
        }

    # Instantiate a svr
    svr = Pipeline(
        [
            ('std', StandardScaler()),
            ('svr', SVR())
        ]
    )

    grid = do_grid_search(svr, param_grid, X, y)

    # save the model, write params to config file
    # write the output
    estimator_name = "svr"
    post_training(estimator_name, grid)


def post_training(estimator_name, grid):
    # do the writing of output paramters, scores,
    # save the model.
    save_model(grid.best_estimator_, estimator_name)
    write_model_config(estimator_name, grid.best_estimator_)
    write_best_params(estimator_name, grid)


def save_model(estimator, estimator_name):
    # Pickle the model
    filename = output_path + estimator_name + ".pkl"
    job = joblib.dump(estimator, filename)
    print(f"{str(estimator)} \n is pickled to {job}")


def do_grid_search(estimator, param_grid, X, y):
    grid = GridSearchCV(
        estimator, param_grid,
        scoring=scoring,
        cv=5,
        refit=scoring
    )
    grid = grid.fit(X, y)
    return grid


def write_model_config(estimator_name, estimator):
    # Write the best params from grid search
    config = ConfigParser()

    # Only get params for this estimator
    params = estimator.get_params(deep=False)

    # Set the config section name
    config[estimator_name] = {}
    # The individual items under this section
    for k, v in params.items():
        config[estimator_name][k] = str(v)

    # write output
    with open(models_conf, 'a') as configfile:
        config.write(configfile)


def write_best_params(estimator_name, grid):

    # Also output the best params to a quick
    # viewing output file

    f = open(output_file, 'a')
    # Best params
    f.write(f"Model {estimator_name}\n")
    f.write(f"{str(grid.best_estimator_)} \n")

    # Use a ConfigParser for easy reading and
    # writing of sections and items.
    # Write the score of each model.
    config = ConfigParser()
    # Write the section name as the model name
    config[estimator_name] = {}
    # Write the individual items under this section
    result = grid.best_score_
    config[estimator_name][scoring] = str(result)
    f.write(f"best_score {str(result)} \n")

    result = grid.best_params_
    for k, v in result.items():
        config[estimator_name][k] = str(v)
        f.write(f"{str(k)} {str(v)} \n")

    cv_result = grid.cv_results_
    for k, v in cv_result.items():
        config[estimator_name][k] = str(v)

    # write output
    with open(cv_score_conf, 'a') as configfile:
        config.write(configfile)

    # Close the output file
    f.close()
