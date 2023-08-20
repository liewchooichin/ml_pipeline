# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:10:51 2023.

@author: Liew

Preprocessing the raw dataframe features and label.
"""

import numpy as np
import pandas as pd
import joblib
from my_std_lib import output_path


def _process_sleep(raw):
    # Cast the sleep_time and wake_time to date time
    format = '%H:%M'
    raw['sleep_time'] = pd.to_datetime(raw['sleep_time'], format=format)
    raw['wake_time'] = pd.to_datetime(raw['wake_time'], format=format)

    # Find the hour of sleep
    diff_time = pd.to_timedelta(raw['wake_time']-raw['sleep_time'], unit='h')

    # Because Timedelta has seconds to work with
    # Convert seconds to hours
    # Add sleep hours to the data
    raw['sleep_duration'] = diff_time.dt.seconds / 3600

    # Binarize the duration
    raw['sleep_enough'] = pd.cut(
        raw['sleep_duration'],
        bins=[0, 7, 10],
        labels=['0', '1'],
        retbins=False
    )
    raw.drop(columns='sleep_time', inplace=True)
    raw.drop(columns='wake_time', inplace=True)

    # Keep the sleep_duration in case it is needed


def _process_attendance(raw):
    # Imputing with mean value
    med = raw['attendance_rate'].agg(np.median)
    raw['attendance_rate'] = raw['attendance_rate'].fillna(value=med)
    # Binarize the attendance_rate
    q1 = raw['attendance_rate'].quantile(0.25)
    raw['attendance_enough'] = pd.cut(
        raw['attendance_rate'],
        bins=[0, q1, 100.0],
        labels=['0', '1'],
        retbins=False
    )


def _process_final_test(raw):
    # Drop the missing values
    raw.dropna(subset=['final_test'], how='any', inplace=True)


def _process_direct_admission(raw):
    # binary encoding
    # all dummies are processed together
    # make the category into lowercase
    raw['direct_admission'] = raw['direct_admission'].apply(str.lower)
    raw['direct_admission'] = raw['direct_admission'].apply(
        lambda s: _replace_zero_one(s, 'no', 'yes'))


def _process_cca(raw):
    # make all the category names into lowercase
    # cca=none remains the same,
    # cca=arts, sports, clubs will be changed to some
    # categorize into none or some
    # make the values into all lowercase
    raw['CCA'] = raw['CCA'].apply(str.lower)
    # replace none or yes for the different sports

    def replace_category(s, new_name, old_name_list):
        if s in old_name_list:
            s = s.replace(s, new_name)
        return s

    # both lines will work
    raw['CCA'] = raw['CCA'].apply(lambda s: replace_category(
        s, 'yes', ['arts', 'sports', 'clubs']))
    raw['CCA'] = raw['CCA'].apply(
        lambda s: _replace_zero_one(s, 'none', 'yes'))
    # rename to lowercase
    raw.rename(columns={'CCA': 'cca'}, inplace=True)


def _process_tuition(raw):
    # Make the unique values into only 'yes', 'no'
    # Make the category values lowercase
    def replace_tuition(s):
        if s in ['Yes', 'Y']:
            s = s.replace(s, 'yes')
        elif s in ['No', 'N']:
            s = s.replace(s, 'no')
        return s

    raw['tuition'] = raw['tuition'].apply(replace_tuition)
    raw['tuition'] = raw['tuition'].apply(
        lambda s: _replace_zero_one(s, 'no', 'yes'))


def _process_learning_style(raw):
    # binary encoding
    # all dummies are processed together
    # make the categor to lowercase
    raw['learning_style'] = raw['learning_style'].str.lower()
    raw['learning_style'] = raw['learning_style'].apply(
        lambda s: _replace_zero_one(s, 'auditory', 'visual'))


def _process_number_of_siblings(raw):
    # Change the number to category string for use
    # with get_dummies.
    def replace_num(s):
        if s == 0:
            return 'zero'
        elif s == 1:
            return 'one'
        elif s == 2:
            return 'two'

    # raw['number_of_siblings'] = raw['number_of_siblings'].apply(replace_num)
    pass  # leave the number as it is


def _process_transport(raw):
    # Change the category values to shorter names

    # This values is actually not used.
    def replace_transport(s):
        if s == 'private transport':
            return 'private'
        elif s == 'public transport':
            return 'public'
        return s  # unchanged for walk

    # if this is used, it can be used to changed to dummies
    # change the category values for later get_dummies
    raw['mode_of_transport'] = raw['mode_of_transport'].apply(
        replace_transport)


def _process_age(raw):
    # Correct the values
    # Original age has mistakes in the input as:
    # The age -5, -4, 5 and 6 appears to be mistakes in the input.
    # The majority of values is 15 and 16.
    # Therefore, it is quite safe to assume that the -5 and 5 are
    # actually 15, and -4 and 6 are actually 16.
    # Use category for age for easy comparison.

    # This feature is actually not used in learning.
    def correct_age(a):
        if (a == -5) or (a == 5) or (a == 15):
            a = 'fifteen'
        elif (a == -4) or (a == 6) or (a == 16):
            a = 'sixteen'
        return a

    raw['age'] = raw['age'].apply(correct_age)


def _replace_zero_one(s, val_zero, val_one):
    # val_zero = the original value to be replaced with 0
    # val_one = the original value to be replaced with 1
    # change all to lowercase
    s = s.lower()
    val_zero = val_zero.lower()
    val_one = val_one.lower()
    if s == val_zero:
        s = 0
    elif s == val_one:
        s = 1
    return s


def _drop_cols(raw, col_names):
    # drop columns that are not used
    for col in col_names:
        raw.drop(columns=col, inplace=True)


def process_all_cols(raw):
    """
    Process all the columns.

    Parameters
    ----------
    raw : pd.DataFrame
        The raw dataframe.

    Returns
    -------
    pd.DataFrame
        The processed dataframe with no null values.
        Ready for learning.

    """
    # process all the columns

    # sleep
    _process_sleep(raw)
    print("sleep_enough")
    print(f"{raw['sleep_enough'].unique()}")

    # attendance_rate
    _process_attendance(raw)
    print("attendance_enough")
    print(f"{raw['attendance_enough'].unique()}")

    # nan in final_test
    _process_final_test(raw)
    no_null = not raw["final_test"].isna().any()
    print("No null values in final_test: "
          f"{no_null}")

    # binary encoding for direct_admission
    _process_direct_admission(raw)
    print("direct_admission is used as it is:")
    print(f"{raw['direct_admission'].unique()}")

    # make cca into only two categories
    _process_cca(raw)
    print("cca")
    print(f"{raw['cca'].unique()}")

    # make tuition into two classes
    _process_tuition(raw)
    print("tuition")
    print(f"{raw['tuition'].unique()}")

    # make learning_style into two categories
    _process_learning_style(raw)
    print("learning_style")
    print(f"{raw['learning_style'].unique()}")

    # change the number of siblings into category string
    _process_number_of_siblings(raw)
    print("number_of_siblings")
    print(f"{raw['number_of_siblings'].unique()}")

    # change the transport category name
    # this feature is not used for learning.
    _process_transport(raw)

    # correct the age value
    # this feature is not used for learning.
    _process_age(raw)

    # drop columns that are not used
    unused = ['gender', 'n_female', 'n_male', 'mode_of_transport',
              'age', 'bag_color', 'student_id', 'index']
    _drop_cols(raw, unused)
    print("Not using these features:")
    for s in unused:
        print(f"- {s}")

    # get dummies for number_of_siblings only
    cols = ['number_of_siblings']
    raw = pd.get_dummies(raw, columns=cols, dtype=np.uint8)
    print("Features for learning:")
    print(f"{len(raw.columns)} total")
    for s in raw.columns:
        print(f"- {s}")

    return raw  # in processed form


def save_dataframe(dataframe, filename="dataframe"):
    """
    Save the dataframe to a pickle file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Save a dataframe for later use. Default dataframe

    filename : str
        A name for the name. Output data path
        and ext .pkl will be appended by default.

    Returns
    -------
    None.

    """
    pickle_name = output_path + filename + '.pkl'
    j = joblib.dump(dataframe, pickle_name)
    print(f"{j} is pickled to {pickle_name}")


# Global variables
# Default filename for the preprocessed file
default_filename = "prep_df"


def prep_load(filename=default_filename):
    """
    Load a pickled dataframe.

    Parameters
    ----------
    filename : str, optional
        Filename of the pickle file. The default is "prep_df".
        The default path is data/prep_df.pkl.

    Returns
    -------
    pred_df : pd.DataFrame
        Dataframe loaded from the pickle file.

    """
    pickle_name = output_path + filename + '.pkl'
    prep_df = joblib.load(pickle_name)
    return prep_df


def prep_save(raw_dataframe, filename=default_filename):
    """
    Preprocessed and save dataframe to a pickle file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Plain, raw dataframe that has not been preprocessed.

    filename : str
        Filename to the pickled dataframe.
        Default is "prep_df".

    Returns
    -------
    None.

    """
    df_prep = process_all_cols(raw_dataframe)
    save_dataframe(df_prep, filename)
