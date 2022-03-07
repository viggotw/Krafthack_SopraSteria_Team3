import pandas as pd
from typing import List

def get_timeslots(df: pd.DataFrame, sampling_rate: int = 1) -> List[pd.DataFrame]:
    ''' Splits a dataframe into chunks with equal sampling rate
    :param df: DataFrame with a datatime index
    :returns: List of dataframes with similar samling rate
    '''
    # Find timestamps where the data does a "jump"
    time_jumps_sec = (df.index[1:] - df.index[:-1]).seconds

    # Find start and end of each timeslot
    timeslots_start = [df.index[0]] + list(df.index[1:][time_jumps_sec>sampling_rate])
    timeslots_end = list(df.index[:-1][time_jumps_sec>sampling_rate]) + [df.index[-1]]

    return [df[ts_start:ts_end].copy() for ts_start, ts_end in zip(timeslots_start, timeslots_end)]


def get_temporal_lookback_features(df: pd.DataFrame, cols: List[str], window_size: int, steps: int = 1) -> List[pd.DataFrame]:
    ''' Cretes new features that contains past information for selected features
    :param df: DataFrame with a datatime index
    :param cols: List of columns to be used
    :param window_size: How far into the past you want to look
    :param steps: Step size for each past row
    :returns: A new dataframe with new "look-back" features 
    '''
    for col in cols:
        for i in range(1, window_size, steps):
            df.loc[:,f"{col}_t-{i}"] = df.loc[:,col].shift(i)

    df = df.loc[df[cols].dropna().index]
    return df


def get_temporal_lookback_df(list_of_dfs: List[pd.DataFrame], cols: List[str], window_size: int, steps: int = 1) -> pd.DataFrame:
    ''' Accepts a list of dataframes, creates look-back features for each of them
    and concatenates the results into a dataframe
    :param list_of_dfs: List of dataframes with a datatime index
    :param cols: Columns to be affected by the look-back calculations
    :param window_size: How far into the past you want to look
    :param steps: Step size for each past row
    :returns: A concatenated dataframe with additional lookback features
    '''
    new_df = []

    for df in list_of_dfs:
        new_df.append(
            get_temporal_lookback_features(df, cols=cols, window_size=window_size, steps=steps)
        )

    return pd.concat(new_df, axis=0)


def add_seconds_operational(dataframe):
    # Find index of "start" modes in the timeseries
    start_ts = dataframe[dataframe['mode'] == 'start'].index
    # Calculate secods until next "start" mode
    secs_since_last_start = (start_ts[1:] - start_ts[:-1]).seconds
    # Extract a list of "timeslots" that pairwise indicate a sequence of operation
    last_start_before_counting = start_ts[:-1][secs_since_last_start > 1]
    last_start_before_counting = list(last_start_before_counting)
    last_start_before_counting.append(dataframe.index[-1])
    # Create new feature
    dataframe['sec_since_last_start'] = 0
    for t1, t2 in zip(last_start_before_counting[:-1], last_start_before_counting[1:]):
        dataframe.loc[t1:t2, 'sec_since_last_start'] = range(len(dataframe[t1:t2]))
    # Force "start" mde to equal 0 seconds
    dataframe.loc[dataframe['mode'] == 'start', 'sec_since_last_start'] = 0
    # Look at the last timeslot
    dataframe.loc[t1:t2, ['mode', 'sec_since_last_start']]
    return dataframe


def add_hour_feature(dataframe):
    dataframe['hour'] = dataframe.index.hour
    return dataframe