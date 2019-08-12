"""
data_processor.py: Utility script for retrieving and storing data in CSV and HDF5 format

__author__ = "Victor Marco Milli"
__version__ = "0.9.1"
__maintainer__ = "Victor Marco Milli"
__status__ = "Project/study script for project SWISS / Bise"

"""
from typing import Any, Union

import constants as const
import random as rand
import data_handler as dh
import pandas as pd


def generate_resampled_df(data, target_col, date_cols, pred_period=const.steps_24_hours, max_len=None):
    """
    Generates a new balanced dataset by randomly replacing negative events
    from the given dataframe, which should be the master used for model training.
    This is achieved by replacing blocks of negative events with blocks of positive events
    from the same df but different year, taking seasonality in account.
    """
    df = data.copy()
    if max_len and max_len < len(data):
        df = df.iloc[:max_len]

    event_indexes = df.index[df[target_col] == 1].tolist()

    yearly_ranges = collect_event_start_end_indexes(df, target_col)

    # following list has 516 periods: CHECK!!
    # event_ranges1 = collect_event_start_end_indexes_0(event_indexes)
    # print(event_ranges1)

    event_ranges = []

    for key1, value1 in yearly_ranges.items():
        for key, value in value1.items():
            # print(key, value)
            event_ranges = event_ranges + value

    # following list has 517 periods: CHECK!!

    # print()

    y = 0
    m = 0
    season = get_season(0)
    block_start = 0
    range_list_padded = []
    for r in event_ranges:
        block_end = r[0] - 1

        y_temp = df.Year.iloc[block_start]
        m = df.Month.iloc[block_start]
        season_temp = get_season(m)

        if y_temp != y or season_temp != season:
            # print('***************')
            # print(season_temp, season, y_temp, y)
            y = y_temp
            season = season_temp
            # print(y, m, block_start, block_end)
            range_list_padded = collect_replacing_random_padded_season_event_ranges(yearly_ranges, y, season, df.Year.unique())
        print(df.shape)
        df = replace_non_events_with_events(df, date_cols, block_start, block_end, range_list_padded)
        block_start = r[1] + 1

    return df


def replace_non_events_with_events(df1, date_cols, block_start, block_end, range_list_padded):
    """This function performs the actual block replacement within the dataset. The given date_cols will not be be replaced.

    The actual version is a fix for a bug in the initial version, which did not  perform any updates. This fix implies that all
    date columns should always be leading, followed by all other columns, which shall be replaced rowwise
    """
    print('Replacing', block_start, block_end)
    df = df1.copy()
    columns = [c for c in df.columns if c not in date_cols]
    #print(columns)
    start = block_start
    i = 0
    redoing = False
    while start < block_end:
        print('Replacing block', i, start, block_end)
        replacement = range_list_padded[i]
        replacement_len = replacement[1] - replacement[0] + 1
        if start + replacement_len - 1 <= block_end:
            sample_data = df.iloc[replacement[0]: replacement[1] + 1][columns].values
            #print(sample_data)
            #print('Shape out', df.iloc[start:start + replacement_len][columns].shape)
            #print('Shape in', df.iloc[replacement[0]:replacement[1]][columns].shape)
            #df.iloc[start:start + replacement_len][columns] = sample_data
            replacement_del = start + replacement_len
            df.iloc[start:replacement_del, len(date_cols):] = sample_data

            start = replacement_del
        if i == len(range_list_padded) - 1 and not redoing:
            i = 0
            redoing = True
        if i == len(range_list_padded) - 1 and redoing:
            return df
        else:
            i += 1
    return df


def get_season(month):
    """In order to produce a realistic sampling, seasonality was taken into account. This function determines
     from which yearly season the replacement data have to be extracted"""
    season_matrix = [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]]
    season = []
    for s in season_matrix:
        if month in s:
            season = s
            break
    return season


def collect_replacing_random_padded_season_event_ranges(yearly_ranges, year, season, years_unique, pred_period=const.steps_24_hours):
    """
    Collects random event ranges from the given season from years other than the provided year.
    The event ranges are indexes of rows from the dataframe corresponding to the beginning
    and end of periods classified as event occurences. The periods were extended with random padding
    corresponding to transition non event period
    """
    range_list = []
    years = [y for y in years_unique if year != y]
    rand.shuffle(years)
    rand.shuffle(season)
    for y in years:
        monthly_per = yearly_ranges.get(y)
        for m in season:
            range_list = range_list + monthly_per.get(m)
    rand.shuffle(range_list)
    # print(range_list)

    range_list_padded = []
    for r in range_list:
        start = r[0] - get_random_padding()
        end = r[1] + get_random_padding()
        range_list_padded.append((start, end))
    # print(padding)
    return range_list_padded


def get_random_padding(pred_period=const.steps_24_hours, min_val=const.steps_12_hours):
    """Generates a random number, not inferior to the provided min_val value. This is used to extend the sampled data
    with a pre- and post-event period"""
    padding = rand.randint(1, 2)
    padding = max(int(rand.random() * padding * pred_period), min_val)
    return padding


def collect_event_start_end_indexes(df, target_col):
    """Collects dataset indexes of rows which correspond to the first occurence of an event period,
    as marked in the provided target_col"""
    yearly_events = dict()

    df_event = df[df[target_col] == 1]

    for y in df.Year.unique():
        df_y = df_event[df_event['Year'] == y]
        month_indexes = dict()
        for m in range(1, 13):
            event_indexes = df_y.index[df_y['Month'] == m].tolist()
            index_ranges = []
            i = 0
            while i < len(event_indexes):

                first_index = event_indexes[i]
                for j in range(i + 1, len(event_indexes)):
                    if j == len(event_indexes) - 1 or event_indexes[j + 1] - event_indexes[j] > 1:
                        last_index = event_indexes[j]
                        period = (first_index, last_index)
                        # print(period)
                        index_ranges.append(period)
                        i = j + 1
                        break
            # print(m, index_ranges)
            month_indexes[m] = index_ranges
        yearly_events[y] = month_indexes

    return yearly_events

def resample(df, output_file, target_col='rule4_episode'):


    #df['Year'] = pd.DatetimeIndex(df['datetime']).year
    #df['Month'] = pd.DatetimeIndex(df['datetime']).month
    df['original_index'] = df.index.values.astype(int)

    print(df.head())

    new_col_order = ['datetime', 'Year', 'Month', 'original_index', target_col]
    new_col_order = new_col_order + [c for c in df.columns if not c in new_col_order]
    print(new_col_order)
    df = df.reindex(columns=new_col_order)
    print(df.head())
    print(df.columns)


    print(df[target_col].value_counts())
    df2 = generate_resampled_df(df, target_col, ['datetime', 'Year', 'Month'], pred_period=const.steps_12_hours)
    print(df2[target_col].value_counts())
    dh.store_df_as_csv_compressed(df2, output_file)

    return df2

def get_station_features(df, station, *additional_cols):
    """Extracts the datetime and columns relevant to the provided weather station,
    plus any additional columns"""
    cols = [c for c in df.columns if c.startswith(station)]
    #print(additional_cols[0])
    if additional_cols  and len(additional_cols) > 0:
        cols = additional_cols[0] + cols
    cols = ['datetime'] + cols
    return df[cols]


def get_dataset_slices(df, col_list1, col_list2):

    return df[col_list1], df[col_list2]


def left_join_datasets(left, right, on='datetime'):
    """
    Performs left join (merge) upon column 'on'.
    At least one of the dataframes should carry the 'on' column
    :param left: dataframe
    :param right: dataframe
    :param on: column upon which the dataframes are merged
    :return: new dataframe
    """

    drop_on_from_left = False
    drop_on_from_right = False

    if not (on in left.columns):
        if not (on in right.columns):
            raise Exception('Specified column to join on in neither dataset')
        else:
            left.insert(0, on, right[on])
            drop_on_from_left = True

    if not (on in right.columns):
        right.insert(0, on, left[on])
        drop_on_from_right = True

    df = pd.merge(left, right, on=on)

    if drop_on_from_left:
        left.drop(columns=[on], inplace=True)
    elif drop_on_from_right:
        right.drop(columns=[on], inplace=True)

    return df


def extract_year_as_column(df, datetime_col='datetime'):
    if not (datetime_col in df.columns):
        raise Exception('Datetime column expected')

    df[column_year] = pd.DatetimeIndex(df[datetime_col]).year


def extract_month_as_column(df, datetime_col='datetime'):
    if not (datetime_col in df.columns):
        raise Exception('Datetime column expected')

    df[column_month] = pd.DatetimeIndex(df[datetime_col]).month


def extract_hour_as_column(df, datetime_col='datetime'):
    if not (datetime_col in df.columns):
        raise Exception('Datetime column expected')

    df[column_hour] = pd.DatetimeIndex(df[datetime_col]).hour




