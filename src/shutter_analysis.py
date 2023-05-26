import src.diamon_analysis as da
import src.influx_data_query as idb
from datetime import datetime
import pandas as pd
import pytz

class beamline():
    def __init__(self, ref, df):
        self.target_station = ref[0]
        if self.target_station == "1":
            self.current_info = "ts1_current"
        elif self.target_station == "2":
            self.current_info = "ts2_current"
        if ref[1] == "B":
            self.get_beam_info(ref)
        else:
            self.get_info(ref, df)

    def __str__(self):
        return self.name

    def get_info(self, ref, df):

        beamline = df[df["key"] == ref[1:3]]
        self.name = beamline.Name.values[0]
        self.building_position = beamline.index.get_level_values("Location")[0]
        beam_df = df.xs(beamline.index.get_level_values("Location")[0], level="Location").set_index("Name")
        self.all_neighbours = beam_df
        self.closest_neighbours = self.set_neighbours(beam_df)
        self.influx_data = self.all_neighbours.index.tolist()
        self.influx_data.append(self.current_info)

    def get_beam_info(self, ref):
        self.name = "no_beamline"
        self.influx_data = [self.current_info]
    def set_neighbours(self, df):
        idx = df.loc[self.name].Number
        return df[(df.Number == idx -1) | (df.Number == idx +1)]
"""
class beamline():
    def __init__(self, key, beam_df):
        self.target_station = key[0]
        self.name = key.Name.values[0]
        self.position = key.index.get_level_values("Location")[0]
        self.beamlines =  beam_df
        self.set_neighbours(beam_df)
    def __str__(self):
        return self.name
    def set_neighbours(self, df):
        idx = df.loc[self.name].Number
        self.neighbours = df[(df.Number == idx -1) | (df.Number == idx +1)]
"""
def get_match_dict(match, dic):
    for key in dic.keys():
        if key == match:
            return dic[key]

def check_updated_shutter_info(shutters, beam_df, selected_param="ts2_current"):
    # add check to load new data
    date = get_date_df(shutters["current"], selected_param)
    if date < datetime.now(pytz.utc):
        print("Updating shutter information \n")
        shutters = append_new_shutter_info(shutters, beam_df)
        #saves new shutter information into pickle for later use
        da.save_pickle(shutters, "shutter_data")
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

def append_new_shutter_info(shutters, beam_df):
    new_data = latest_shutters(shutters, beam_df)
    new_shutters = {key: pd.concat([shutters["shutters"][key], df]) for key, df in new_data["shutters"].items()}
    new_current = {key: pd.concat([shutters["current"][key], df]) for key, df in new_data["current"].items()}
    return {"shutters": new_shutters, "current":new_current}

def latest_shutters(current_shutter, beam_df):
    last_time = get_date_df(current_shutter["current"],"ts2_current")
    last_time = last_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    today = idb.date_to_str(datetime.today())
    #today = datetime.today().isoformat()
    dates = [last_time, today]
    shutters = influx_db_query(dates, beam_df.channel_name)
    return shutters

def influx_db_query(dates, names=None):
    """"
    load influx database between selected dates and option include specific channel names
    args:
        dates (list of datetime as STR)
        names (optiponal): str list of channel names to query if none select all beamline and current info
    """
    query_obj = idb.query_object()
    query_obj.start = dates[0]
    query_obj.end = dates[1]
    if names is None:
        query_obj.names = idb.channel_names()
    else:
        query_obj.names = names
    query_data = query_obj.influx_query()
    return query_data

def filter_shutters(influx_info, start, time, names):
    values = [df[df.index.get_level_values("datetime") < time].groupby(names[i]).tail(1) for i, df in enumerate(influx_info)]
 
    return values