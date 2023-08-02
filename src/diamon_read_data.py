# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:55:42 2022

This file will process both the F_UNFOLD and the rate data from DIAMON spectrometer
"""
import pandas as pd
import glob
from datetime import datetime
from pathlib import Path
import src.neutronics_analysis as na
import src.diamon as diamon
import src.beamline as b

loc_path = "data\measurement_location.csv"
target_station_path = "data\target_station_data.csv"
diamon_path = "data\measurements\DIAMON*"

# read all data
def read_data():
    # try to load exisiting data
    loc_data = na.read_csv(loc_path, {'x': float, 'y': float, 'z':float})
    try:
        data = na.load_pickle("diamon_data")
        print("checking if there is new data in the directory")
        data = update_diamon_data(data, loc_data)
    except FileNotFoundError:
        print("No existing diamon data- loading all data in file path")
        data = load_diamon(loc_data)
    return data

def update_diamon_data(existing_data, loc_data):
    """_summary_

    Args:
        existing_data (_type_): _description_
        loc_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # check if missing
    latest_data = get_last_entry(existing_data)
    file_paths = glob.glob(r"data\ts2_measurements\DIAMON*\counters.txt")
    # gets all paths that haven't been read into the dictionary of data
    unread_paths = [
        x for file_path in file_paths if (x := read_unloaded_file(file_path, latest_data)) is not None]
    if len(unread_paths) == 0:
        print("No new data - loading existing file")
        return existing_data
    else:
         new_dic = load_diamon(loc_data, unread_paths)
         existing_data.update(new_dic)
         return new_dic

def get_last_entry(data):
    # in exisitng group for diamon results get most recent entry
    return max([file.start_time for file in data.values()])

def get_earliest_entry(data):
    # in exisitng group for diamon results get most recent entry
    return min([file.start_time for file in data.values()])

def read_unloaded_file(file_path : str, latest_data : datetime):
    """
    checks whether a file is from after the latest loaded entry
    Args:
        file_path (str): path to file
        latest_data (datetime): datetime of latest entry

    Returns:
        _type_: _description_
    """
    # record names of files to read in
    lines = na.read_lines(file_path)
    for i, line in enumerate(lines):
        if "GMT time" in lines[i-1]:
            # convert to correct GMT time
            aware_str = na.str_to_local_time(line)
            if aware_str > latest_data:
                return Path(file_path).parent
            else:
                return None

# update info
def load_diamon(loc_data : pd.DataFrame, data_path : str = diamon_path):
    """data from diamon instrument loaded from files. data matched with measurement coordinate
    Args:
        loc_data (df): pandas dataframe of x, y, z coordinates and data reference
        fname (str): folder directory to diamon data

    Returns:
        dictionary of diamon class objects
    """
    diamon_list = [diamon.diamon(folder) for folder in glob.glob(data_path)]
    # match the location from the file name ref to coord to get shutter info
    diamon_list = match_id_location(loc_data, diamon_list)
    diamon_dict =  {data.file_name: data for data in diamon_list}
    return diamon_dict

def match_id_location(loc_data : pd.DataFrame, data : dict[diamon.diamon]):
    """
    from location data and id & start of measurement match the id and coordinates
    Args:
        loc_data (pd.DataFrame): csv of loc id and coordinate
        data (dict[diamon]): dictionary of diamon instance containing all data

    Returns:
        dict: dictionary with diamon data all matched to a location
    """
    id_list = [diamon_obj.id for diamon_obj in data]
    id = get_measurement_id(pd.DataFrame(id_list))
    data = [match_location(loc_data, diamon_obj, id) for diamon_obj in data]
    return data

def get_measurement_id(df : pd.DataFrame):
    """
    creates a key from the start, and id of the measurement. this contains the date and
    # of which the measurement was at for that day
    Args:
        df (pd.DataFrame): 

    Returns:
        df (pd.DataFrame) : sorted df in order of keys
    """
    sorted_df = df.sort_values(by="start").reset_index().drop_duplicates(subset=["start"], keep="first", ignore_index=True)
    sorted_df["count"] = sorted_df.groupby(by=[sorted_df['start'].dt.date, sorted_df["serial"]]).cumcount()+1
    sorted_df["key"] = ((sorted_df["start"].dt.strftime("%d.%m.")) + (sorted_df["count"].astype(str)) + "-" + sorted_df["serial"])
    return sorted_df

def match_location(location_data, data, id):
    """match location csv file key to data file name

    Args:
        location_data (df): pd df of xyz and ref.
        diamon_list (list): list of diamon class obj
        series_list (list): list of pd data series containing file id's

    Returns:
        dic: list of diamon obj with keys assigned
    """
    # first check there isnt a duplicate
    # load location
    data.file_name = id[id["start"] == data.start_time]["key"].values[0]

    #what if cant find reference
    data.reference = location_data.loc[location_data["Name"] == data.file_name].reset_index()
    #find 2d distance
    data.find_distance(2)
    #get beamline and building names for data being measured
    beamline_df = pd.read_csv("data/target_station_data.csv", index_col=["Building", "Location"])
    data.beamlines = b.beamline(data.reference["Measurement Reference"].iloc[0], beamline_df)
    return data

if __name__ == '__main__':
    data = read_data()
