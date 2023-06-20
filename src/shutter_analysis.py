import src.diamon_analysis as da
import src.influx_data_query as idb
import src.neutronics_analysis as na
from datetime import datetime
import pandas as pd
import pytz
import numpy as np

class beamline():
    def __init__(self, ref, df):
        self.target_station = ref[0]
        if self.target_station == "1":
            self.current_info = "ts1_current"
        elif self.target_station == "2":
            self.current_info = "ts2_current"
        self.get_info(ref, df)

    def __str__(self):
        return self.name

    def get_info(self, ref, df):
        beam_info = df[df["key"] == ref[1:3]]
        self.name = beam_info.Name.values[0]
        self.set_building_position(beam_info)
        beam_df = df.xs(beam_info.index.get_level_values("Location")[0], level="Location").set_index("Name")
        self.all_neighbours = beam_df
        self.closest_neighbours = self.set_neighbours(beam_df)
        self.influx_data = self.all_neighbours.index.tolist()
        self.influx_data.append(self.current_info)

    def get_beam_info(self):
        self.name = "no_beamline"
        self.influx_data = [self.current_info]

    def set_neighbours(self, df):
        idx = df.loc[self.name].Number
        return df[(df.Number == idx -1) | (df.Number == idx +1)]

    def set_building_position(self, beamline):
        self.building_position = beamline.index.get_level_values("Location")[0]
    @staticmethod
    def get_location(beam, df):
        try:
            location = beam.location
        except AttributeError:
            beamline.set_location_result(beam, df)
            location = beam.location
        except KeyError:
            print("no beamline")
        return location
def load_shutter_data():
    shutter_data = na.load_pickle("shutter_data")
    beamline_df = pd.read_csv("data/target_station_data.csv", index_col=["Building", "Name"])
    updated_shutters = da.check_updated_shutter_info(shutter_data, beamline_df)
    return updated_shutters

def get_match_dict(match, dic):
    for key in dic.keys():
        if key == match:
            return dic[key]

def check_updated_shutter_info(shutters, beam_df, selected_param="ts2_current"):
    # add check to load new data
    date = get_date_df(shutters, "ts2_current", "_time")
    if date < datetime.now(pytz.utc):
        print("Updating shutter information \n")
        new_data = latest_shutters(shutters, beam_df)
        shutters = append_new_shutter_info(shutters, new_data)
        #saves new shutter information into pickle for later use
        da.save_pickle(shutters, "shutter_data")
        print("saved new data and loaded into program \n")
    return shutters

def get_date_df(df, channel_name, max=True):
    """This function extracts in a series of datetime the last date/ first date 
    Args:
        df (pandas dataframe): df contianing time series column
        channel_name: str name of beamline shuttter/current
        colname: str of name of column
    Returns:
        datetime
    """
    time = df[channel_name].index
    if max:
        last_time = time.max()
    else:
        last_time = time.min()
    return last_time

def append_new_shutter_info(shutters, new_data):
    new_shutters = {key: pd.concat([shutters[key], df])  if key in shutters.keys() else shutters[key] for key, df in new_data.items()}
    #get correct names
    shutters = channel_names(new_shutters)
    return shutters

def latest_shutters(current_shutter, beam_df):
    last_time = get_date_df(current_shutter["current"],"ts2_current")
    last_time = last_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    today = idb.date_to_str(datetime.today())
    dates = [last_time, today]
    shutters = influx_db_query(dates, beam_df.channel_name)
    return shutters

def get_initial_date(start, shutters):
    beam_df = pd.read_csv("data/target_station_data.csv", index_col=["Building", "Name"])
    end = get_date_df(shutters, "ts2_current", False)
    end = end.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    start = datetime(2022, 10, 1)
    start = idb.date_to_str(start)
    shutter = influx_db_query([start, end], beam_df.channel_name, update=False)
    comb_shutter = append_new_shutter_info(shutter, shutters)
    return comb_shutter

def influx_db_query(dates, names=None, update=True):
    """"
    load influx database between selected dates and option include specific channel names
    args:
        dates (list of datetime as STR)
        names (optiponal): df of beam info and channel names
    """
    query_obj = idb.query_object(names, True)
    query_obj.start = dates[0]
    query_obj.end = dates[1]
    query_data = query_obj.influx_query(names, update)
    return query_data

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
    shutter_list = {name: df for name, df in shutter_data.items() if name in beamlines}
    times = np.array(data.out_data["datetime"])
    # if on the epb no shutter info - not near a beamline only get current
    for name, df in shutter_list.items():
        df = df.sort_index()
        data.out_data[name] = [get_query_info(df, time, name) for time in times]
    if data.beamlines.name != "no_beamline":
        data.out_data["shutter-open"] = data.out_data[data.beamlines.name]
    #normalise dose to the current
    data = normalise_dose(data)
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
    status = data.loc[:time].tail(1)["_value"].values[0]
    if "current" in name:
        return status
    if (status == 1):
        return True
    else:
        return False

def normalise_dose(data):
    """
    normalise dose measurement to the current of beam
    Args:
        data (obj): diamon class

    Returns:
        data: diamon class object with normalised dose for each time
    """
    if data.beamlines.target_station == "1":
        data.out_data["norm_dose"] = (140 * data.out_data["H*(10)r"].divide(data.out_data[data.beamlines.current_info]).replace(np.inf, 0))
    elif data.beamlines.target_station == "2":
        data.out_data["norm_dose"] = (35 * data.out_data["H*(10)r"].divide(data.out_data[data.beamlines.current_info]).replace(np.inf, 0))
    data.norm_dose = data.out_data["norm_dose"].tail(1)
    return data

#ts1 contains overllaping shutter for different beamlines so need map to correct names
def channel_names(shutter_data):
    names_df = pd.read_csv("data/target_station_data.csv", index_col=["Name"])
    data = {}
    for key, channel in shutter_data.items():
        names = names_df[names_df.channel_name == key].index.to_numpy()
        if "current" in key:
            data[key] = channel
        else:
            for name in names:
                data[name] = channel
    return data
