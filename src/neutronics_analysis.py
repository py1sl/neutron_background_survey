import pandas as pd
import re
import pytz
import dill as pickle
import numpy as np
from datetime import datetime

class units:
    def __init__(self):
        self.dose_rate = r"$\mu Sv h^{-1} $"
        self.phi = r"cm$^{-2}$s$^{-1}$"
        self.current = r"$\mu$ A"
        self.dose = r"$\mu Sv$"
        self.norm_dose_rate_ts1 = r"$\frac{\mu Sv h^{-1}}{140 \mu A}$"
        self.norm_dose_rate_ts2 = r"$\frac{\mu Sv h^{-1}}{35 \mu A}$"
        self.energy_read = r""

def read_csv(path, type_dict=object):
    return pd.read_csv(path,  dtype=type_dict)

def date_to_str(current_tz: pytz.tzinfo.BaseTzInfo, target_tz: pytz.tzinfo.BaseTzInfo, date_obj):
    #  normalisation of timezone to target timezone
    localised_dt = localise_datetime(current_tz, target_tz, date_obj)
    date_str = localised_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    return date_str

def localise_datetime(source: pytz.tzinfo.BaseTzInfo, target: pytz.tzinfo.BaseTzInfo, datetime_object):
    """
    Localise datetime object to required timezone
    """
    normalised_datetime = source.normalize(source.localize(datetime_object))
    localised_timezone_datetime = normalised_datetime.astimezone(target)
    localised_plain_datetime = localised_timezone_datetime.replace(tzinfo=None)
    return localised_plain_datetime

def save_pickle(data, name):
    """
    Saves pickled data to file. (less memory)
    """
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(name):
    """
    load pickled data from file into program
    """
    with open(name, 'rb') as f:
        return pickle.load(f)

def str_to_local_time(test_string):
    # converts to local time (london)
    test_date = datetime.strptime(test_string, "%a %b %d %H:%M:%S %Y")
    aware_str = pytz.timezone("Europe/London").localize(test_date)
    return aware_str

def read_lines(path):
    # return list of all lines in a txt file
    with open(path, "r") as f:
        return f.read().splitlines()

#set of functions for string cleaning
def clean_param(line, uncert=None):
    """_summary_

    Args:
        line (_type_): _description_
        uncert (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    line = line.split(":")[1]
    line = clean(line)
    if uncert:
        #uses regular expression package to extract an uncertainty found between a bracket and % symbol
        line[3] = re.findall('\((.*?)%\)', line[3])
        return float(line[0]), float(line[3][0])
    else:
        return float(line[0])

def clean(line):
    line = line.strip()
    line = " ".join(line.split())
    line = line.split()
    return line

def calc_bin_widths(bins: np.array):
    """
        Calculates bin widths for set of bins that is an numpy array
    """
    bw = np.diff(bins)
    bw = np.insert(bw, 0, bins[0], axis=0)
    return bw

def get_match_dict(match, dic):
    """match a dict key to str key

    Args:
        match (str): 
        dic (dict): 

    Returns:
        dict: dict with matching key
    """
    for key in dic.keys():
        if key == match:
            return dic[key]

def sum_columns(df : pd.DataFrame, columns :list[str]):
    """sums all column values on a given axis for given column

    Args:
        df (pd.DataFrame): _description_
        columns (list[str]): _description_
        axis (int) : opt parameter to specify sum on row or column
    Returns:
        df: df with summs
    """
    return df.loc[:, columns].sum(axis=1, skipna=True)