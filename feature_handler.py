"""
feature_handler.py: Utility script for dynamic handling of featuresß

__author__ = "Victor Marco Milli"
__version__ = "0.9.1"
__maintainer__ = "Victor Marco Milli"
__status__ = "Project/study script for project SWISS / Bise"

"""
import pandas as pd
import numpy as np
import re

def get_stations(df, drop_stations=None):

    stations = []
    for c in df.columns:
        x = re.search('^[A-Z]{3}_[A-Z]{1}', c)
        if x:
            if not(drop_stations) or not(x.group() in drop_stations):
                x = re.search('^[A-Z]{3}', x.group())
                print(x.group())
                stations.append(x.group())

    stations = np.unique(stations)
    stations.sort()
    return stations


def get_station_combinations(df, drop_stations=None):

    combinations = []
    stations = get_stations(df, drop_stations)
    for i in range(len(stations)):
        for j in range(i, len(stations)):
            if i != j:
                if not(drop_stations) or not(stations[i] in drop_stations) and not(stations[j] in drop_stations):
                    comb = (stations[i], stations[j])
                    combinations.append(comb)

    return combinations


def calc_feature_diff_between_stations(df, feat_suffix, combinations=None, drop_stations=None, lag=0):

    df1 = df.copy()
    if not(combinations):
        combinations = get_station_combinations(df1, drop_stations)
        print(combinations)

    df2 = df1

    for comb in combinations:

        if not(drop_stations) or not(comb[0] in drop_stations) and not(comb[1] in drop_stations):

            feat1 = comb[0] + '_' + feat_suffix
            feat2 = comb[1] + '_' + feat_suffix
            new_col_name = feat_suffix + '_' + comb[0] + '_minus_' + comb[1]

            fillnas = []

            if lag:

                new_col_name += '_' + str(lag)
                df2 = df.copy()

                """ for i in range(lag):

                    a = df1[feat1].iloc[i:i + 1]
                    b = df2[feat2].iloc[0:1]
                    
                    print(a,b)

                    fillnas.append(a-b) #"""

                df2 = df2.shift(lag)
                #print(len(fillnas))

            print(new_col_name, feat1, feat2)

            df1[new_col_name] = df1[feat1] - df2[feat2]

            """
            i = 0;
            for val in fillnas:
                print(type(val))
                df1.loc[i:i + 1, new_col_name] = val
            """

    return df1


def sort_column(df):

    cols_qnh = [c for c in df.columns if re.search('^QNH_', c)]
    cols_qnh.sort()
    cols_klo = [c for c in df.columns if re.search('^KLO_', c)]
    cols_klo.sort()
    cols_other = [c for c in df.columns if not (c in cols_qnh) and not (c in cols_klo) and c != 'datetime']
    cols_other.sort()
    cols_qnh_klo = [c for c in cols_qnh if re.search('KLO', c)]
    cols_qnh_rest = [c for c in cols_qnh if not (c in cols_qnh_klo)]

    cols_sorted = ['datetime'] + cols_klo + cols_other + cols_qnh_klo + cols_qnh_rest
    print('Number of columns', len(cols_sorted))

    return cols_sorted


