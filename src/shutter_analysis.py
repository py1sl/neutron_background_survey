import src.diamon_analysis as da
import src.influx_data_query as idb
import src.neutronics_analysis as na
from datetime import datetime
import pandas as pd
import numpy as np

def load_channel_data(start : str, end : str, channel_names : list[str]):
    """
    loads channel data from already saved pickle file/influxDB
    Args:
        start(str) : start of measurements
        end(str) : end of measurements
    Returns:
        dict: dict of df for each channel
    """

    try:
        old_channel_data = na.load_pickle("shutter_data")
        channel_data = check_updated_shutter_info(old_channel_data, channel_names, end)
        
    except FileNotFoundError:
        print("No existing channel data saved")
        # will read in all information
        channel_data = idb.query_object.get_data_datetime(start=start, end=end, channels=channel_names)
    channel_data = get_channel_names(channel_data)
    return channel_data

def check_updated_shutter_info(shutters, channel_names, end):
    # add check to load new data
    date = get_date_df(shutters, "local::beam:target")
    if date.date() < end:
        print("Updating shutter information \n")
        shutters = append_new_shutter_info(shutters, channel_names)
        #saves new shutter information into pickle for later use
        na.save_pickle(shutters, "shutter_data")
        print("saved new data and loaded into program \n")
    return shutters

def get_date_df(df, channel_name):
    """This function extracts in a series of datetime the last date
    Args:
        df (pandas dataframe): df contianing time series column
        channel_name: str name of beamline shuttter/current
        colname: str of name of column
    Returns:
        datetime
    """
    time = df[channel_name].index
    last_time = time.max()
    return last_time

def append_new_shutter_info(shutters, channel_names):
    # this function joins new shutter df with existing shutter df/creates a new
    # channel for a new df
    new = latest_shutters(shutters, channel_names).filtered_df
    result = {}
    for key, df in new.items():
        if key in shutters.keys():
            result[key] = pd.concat([shutters[key], df])
        else:
            result[key] = df
    return result

def latest_shutters(current_shutter, channel_names):
    #obtain newest shutter information
    last_time = get_date_df(current_shutter,"local::beam:target")
    last_time = last_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    today = idb.date_to_str(datetime.today(), "Europe/London")
    shutters = idb.query_object.get_data_datetime(last_time, today, channel_names)
    return shutters

def filter_shutters(data, shutter_data):
    """
    filter all shutter query info for beamline shutters and current

    Args:
        data (object): diamon data
        shutters (dict): shutter df

    Returns:
        dict: diamon data with out data having a current and shutter status at each timestep
    """
    #get beamline and building names for data being measured
    beamlines = data.beamlines.influx_data
    print("Filtering data located on " + data.beamlines.name + ", for reference " + data.file_name)
    shutter_list = {name: df for name, df in shutter_data.items() if name in beamlines}
    times = np.array(data.out_data["datetime"])
    # if on the epb no shutter info - not near a beamline only get current
    for name, df in shutter_list.items():
        df = df.sort_index()
        data.out_data[name] = [get_query_info(df, time, name) for time in times]
    if "beamline" not in data.beamlines.name:
        data.out_data["shutter-open"] = data.out_data[data.beamlines.name]
    #normalise dose to the current
    data = da.normalise_dose(data)
    return data

def get_query_info(data, time, name):
    """
    get shutter status at most recent time since the selected time
    Args:
        data (df): shutter df indexed by datetime
        time (datetime object): time of recordingdatetime

    Returns:
        boolean: true if shutter open (2), false if shutter closed/setup (1/3)
    """
    #extract the tail of df where shutter df matches previous times
    try:
        status = data.loc[:time].tail(1)["_value"].values[0]
        return set_shutter_bool(status, name)
    except IndexError:
        # need to get the next time then get opposite
        status = data.loc[time:].iloc[0][0]
        return not set_shutter_bool(status, name)

def set_shutter_bool(status, name):
    # set shutter value (1/2/3) to true/false
    # NEED TO ADD TO OTHER VALUES
    if "current" in name:
        return status
    if (status == 1):
        return True
    else:
        return False

def get_channel_names(shutter_data):
    #ts1 contains overllaping shutter for different beamlines so need map to correct names
    names_df = pd.read_csv("data/target_station_data.csv", index_col=["Name"])
    data = {}
    for key, channel in shutter_data.items():
        names = names_df[names_df.channel_name == key].index.to_numpy()
        if "local::beam:target" == key:
            data["ts1_current"] = channel
        elif "local::beam:target2" == key:
            data["ts2_current"] = channel
        else:
            for name in names:
                data[name] = channel
    return data
