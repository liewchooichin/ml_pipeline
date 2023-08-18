# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:26:55 2023.

@author: Kang Liew Bei

Model prediction and return scores.
"""

from my_std_lib import output_path
from my_std_lib import scoring
from my_std_lib import RND_NUM
from my_prepare_dataset import make_numpy
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error


def get_Xy(dataframe, feature_names, label_name):
    """
    Get numpy array of dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe with features and a label.
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
    X, y = make_numpy(dataframe, feature_names, label_name)

    return X, y


def load_model(estimator_name):
    """
    Load model from pickle file.

    Parameters
    ----------
    estimator_name : str
        Name of the model. The model is stored in path/name.pkl
        of the model.

    Returns
    -------
    model : estimator
        Estimator loaded.

    """
    # default path of the file
    model_file = output_path + estimator_name + ".pkl"

    # use joblib to joad the model
    model = joblib.load(model_file)
    print(f"{model_file} loaded")

    return model


def get_prediction_score(model, X_test, y_test):
    """
    Get prediction score from test set.

    Parameters
    ----------
    model : estimator
        Estimator.
    X_test : ndarray of shape (nsamples, nfeatures)
        Features test set.
    y_test : ndarray of shape (nsamples, )
        Label test set.

    Returns
    -------
    score : dict
        Store multiple scorings of predictions.

    """
    # to store the prediction scorings
    score = dict()

    pred = model.predict(X_test)

    # Loss output is non-negative floating point.
    # The best value is 0.0.
    score['mean_abs_err'] = mean_absolute_error(y_true=y_test, y_pred=pred)

    score['median_abs_err'] = median_absolute_error(y_true=y_test, y_pred=pred)

    score['mean_sq_err'] = mean_squared_error(y_true=y_test, y_pred=pred)

    return score


def get_importance(model, X, y, model_name=None):
    """
    Get the importances of features used by a model.

    Parameters
    ----------
    model_name : str
        Name of the model. If a name is provided, it
        will be loaded.
        If it is None, specify your model (estimator)
        in model.
    model: model that has been loaded
        This model has been loaded and will be used.
    X : ndarray of shape(nsamples, nfeatures)
        Features of the dataset.
    y : ndarray of shape (nsamples, )
        Label of the dataset.

    Returns
    -------
    impt : dict
        Importances of each features.

    """
    # Load a model if the name is provided
    if model_name is not None:
        model = load_model(model_name)

    # Call the actual function to permute
    # importances
    impt = permutation_importance(
        model, X, y,
        scoring=scoring,
        n_repeats=5,
        max_samples=1.0,
        random_state=RND_NUM
    )

    return impt
