# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:36:11 2023

@author: Kang Liew Bei

Read from the score.db.
Then, save the dataframe to a pickle file.
"""

from my_std_lib import data_path
from my_get_db import db_pickle
import os


if __name__ == '__main__':

    # check if the file directory exists
    if os.path.exists(data_path) == False:
        os.mkdir(data_path)

    # URL of db file
    db_url = 'https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db'
    table_name = 'score'
    db_name = data_path + table_name + '.db'
    pickle_name = data_path + 'score.pkl'

    # Store the dataframe in a pickle file
    status = db_pickle(db_url, table_name, db_name, pickle_name)
    print(f"db read in and stored: {status}")
