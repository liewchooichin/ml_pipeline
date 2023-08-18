# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:18:39 2023

@author: Liew

Reading sqlite3 database file from a url,
write the database bytes to a local file,
then use sqlite3 to read the file
and return a dataframe
"""

from my_std_lib import data_path
import pandas as pd
import requests
import sqlite3
import joblib
import os


def db_pickle(db_url, table_name, local_db_name=None, pickle_name=None):
    """
    Read sqlite3 db from the url. Then, store it to a pickle file.

    Parameters
    ----------
    db_url : str
        The url of the sqlite3 db.
    table_name : str
        The name of the database table.
    local_db_name : str
        Local filename to store the db in the url.
    pickle_name : str
        Local filename of the dataframe pickle.

    Returns
    -------
    Status of execution.

    """
    # Default name of the dataframe pickle
    if pickle_name is None:
        pickle_name = table_name + '_df.pkl'

    # Read from the url
    df_data = read_sqlite3_from_url(db_url, table_name, local_db_name)

    # Save the dataframe to a pickle file
    job_done = joblib.dump(df_data, pickle_name)
    print(f"pickle at {job_done}")
    if os.path.isfile(pickle_name):
        return True
    else:
        return False


def read_sqlite3_from_url(db_url, table_name, local_filename=None):
    """
    Read sqlite3 db file from a url and return a dataframe.

    Parameters
    ----------
    db_url : str
        The url like https://---/some.db.
    table_name : str
        The name of the database table.
    local_filename: str
        Local filename to store the db in the url.

    Returns
    -------
    dfa : pd.DataFrame
        pd DataFrame of the db provided in the url.

    """
    # make a request to the url to get the database
    r = requests.get(db_url)
    r.raise_for_status()
    print(f"Read the url of the database in {type(r)}.")
    print(f"Response code is {r.status_code}.")

    print(f"Length of content is {len(r.content)} \n"
          f"of type {type(r.content)}")

    # write the bytes to a local file,
    # the bytes are specified with wb
    # default local filename is table.db
    if local_filename is None:
        local_filename = table_name + '.db'

    with open(local_filename, "wb") as f:
        f.write(r.content)
    f.close()

    # connect the written file to sqlite3
    conn = sqlite3.connect(local_filename)

    # assign the connection to a pandas data frame
    query = "SELECT * FROM " + table_name
    dfa = pd.read_sql(query, conn)

    # close the database connection
    conn.close()
    # remove the temporary file
    # os.remove(local_filename)

    # return the database read
    return dfa


def read_sqlite3_from_file(table_name, local_filename):
    """
    Read a local sqlite3 db file and return a dataframe.

    Parameters
    ----------
    table_name : str
        Table name of the sqlite3 database.
    local_filename : str, optional
        Filename of the db file

    Returns
    -------
    dfa : pd.DataFrame
        Convert the db into a dataframe.

    """
    # check if filename exists
    if os.path.exists(local_filename) is False:
        raise FileNotFoundError(f'{local_filename} not found.')

    # get connection
    conn = sqlite3.connect(local_filename)

    # assign the connection to a pandas data frame
    query = "SELECT * FROM " + table_name
    dfa = pd.read_sql(query, conn)

    # close the database connection
    conn.close()
    # return the database read
    return dfa


def write_df_to_sql(dataframe, table_name):
    """
    Write a dataframe to sqlite3 db.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to be written.
    local_filename : str
        Filename of the db file.
    Returns
    -------
    None.

    """
    local_filename = data_path + table_name + '.db'

    conn = sqlite3.connect(local_filename)
    dataframe.to_sql(table_name, conn, if_exists='replace')

    # close the connection
    conn.close()

    print(f"{local_filename} - sqlite3 db is saved.")
