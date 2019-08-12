"""
data_handler.py: Utility script for retrieving and storing data in CSV and HDF5 format

__author__ = "Victor Marco Milli"
__version__ = "0.9.1"
__maintainer__ = "Victor Marco Milli"
__status__ = "Project/study script for project SWISS / Bise"

"""

import pandas as pd
import os
import os.path



DATA_PATH = '../01_data'

MASTER_DATASET = 'model_data_all_features_extended.csv.zip'

def load_df_from_csv(csv_name):

    df = pd.read_csv(os.path.join(DATA_PATH, csv_name), index_col=0)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

def store_df_as_csv_compressed(df, csv_name):
    df.to_csv(os.path.join(DATA_PATH, csv_name), compression='zip')

def store_df_as_csv(df, csv_name):
    df.to_csv(os.path.join(DATA_PATH, csv_name))

def store_df_as_h5(df, h5_name):
    store = pd.HDFStore(os.path.join(DATA_PATH, h5_name))
    store['df'] = df  # save it
    store.close()

def load_df_from_h5(h5_name):
    return pd.read_hdf(os.path.join(DATA_PATH, h5_name))

def retrieve_file_list(extension=None):

    csvfiles = [f for f in os.listdir(DATA_PATH) if (os.path.isfile(os.path.join(DATA_PATH, f)))]
    if extension:
        csvfiles = [f for f in os.listdir(DATA_PATH) if (os.path.isfile(os.path.join(DATA_PATH, f)) & f.endswith('.'+ extension) )]
    return csvfiles

def retrieve_master_dataset():

    print('Loading master data set:', MASTER_DATASET)
    return load_df_from_csv(MASTER_DATASET)

def master_dataset():

    return MASTER_DATASET

