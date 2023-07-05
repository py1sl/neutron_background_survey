import pandas as pd
import re
import pytz
import dill as pickle
import numpy as np
from datetime import datetime

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
    test_date = datetime.strptime(test_string, "%a %b %d %H:%M:%S %Y")
    aware_str = pytz.timezone("Europe/London").localize(test_date)
    return aware_str

def read_lines(path):
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

def clean_counts(line):
    line = clean(line)
    return int(line[1]), int(line[3])

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