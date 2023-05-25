import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import glob
import pytz
import src.diamon_analysis as da
import src.neutronics_analysis as na
import src.shutter_analysis as sa
import re
import os.path

class diamon:
    def __init__(self, summary_out_selected: bool):
        self.out_data : pd.DataFrame()
        self.rate_data : pd.DataFrame()
        self.datetime : datetime
        self.file_name = ""
        self.pos = []
        self.location = [0,0,0]
        self.energy_bin = []
        self.flux_bin = []
        self.dose : float
        self.unfold_data = pd.Series(dtype=object)
        self.shutters = []
        self.beamlines = []
        self.energy_type = "low"
        self.time = None
        self.start_time = None
        self.end_time = None
        self.id = None
        self.summary = summary_out_selected
        self.current_data = []

    def __str__(self):
        return self.file_name

    def read_counters(self, path):
        with open(path) as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                if "Measurement Time" in lines[i-1]:
                    self.time = " ".join(line.strip().split())
                elif "GMT time" in lines[i-1]:
                    start_time = datetime.strptime(line, "%a %b %d %H:%M:%S %Y")
                    self.start_time = pytz.timezone("Europe/London").localize(start_time)
                    self.end_time = start_time + timedelta(seconds=float(self.time))
                elif "Serial" in line:
                    serial = re.findall(r"\d$", line)[0]
        f.close()
        self.id = pd.Series({"start":self.start_time, "serial":serial})
 
    def read_unfold_file(self, path):
        """Read diamon c/f unfold extracting energy distributions

        Args:
            path (_type_): _description_
        Returns:
            _type_: _description_
        """
        spect_data = False
        with open(path) as f:
            for line in f:
                if " thermal" in line:
                    self.unfold_data["thermal"] = clean_param(line)
                elif "epi" in line:
                    self.unfold_data["epi"] = clean_param(line)
                elif "fast" in line:
                    self.unfold_data["fast"] = clean_param(line)
                elif "phi" in line:
                    self.unfold_data["phi"], self.unfold_data["phi_uncert"] = clean_param(line, True)
                elif "H*(10)_r" in line:
                    self.unfold_data["dose_rate"],  self.unfold_data["dose_rate_uncert"] = clean_param(line, True)
                elif "h*(10)" in line:
                    self.unfold_data["dose_area_product"], self.unfold_data["dose_area_product_uncert"] = clean_param(line, True)
                elif "D1" in line:
                    self.unfold_data["count_D1"], self.unfold_data["count_R"] = clean_counts(line)
                elif "D2" in line:
                    self.unfold_data["count_D2"], self.unfold_data["count_RL"] = clean_counts(line)
                elif "D3" in line:
                    self.unfold_data["count_D3"], self.unfold_data["count_FL"] = clean_counts(line)
                elif "D4" in line:
                    self.unfold_data["count_D4"], self.unfold_data["count_F"] = clean_counts(line)
                elif "D5" in line:
                    self.unfold_data["count_D5"], self.unfold_data["count_FR"] = clean_counts(line)
                elif "D6" in line:
                    self.unfold_data["count_D6"], self.unfold_data["count_RR"] = clean_counts(line)
                elif "TIME" in line:
                    self.unfold_data["time"] = float(clean_param(line))
                elif spect_data and ("----" not in line):
                    line = clean(line)
                    if len(line) < 1:
                        break
                    self.energy_bin.append(float(line[0]))
                    self.flux_bin.append(float(line[-1]))
                elif "Ec" and "Phi(E)*Ec" in line:
                    spect_data = True
        f.close()
        if re.findall(r"[^\/]*C_unfold[^\/]*$", path):
            self.energy_type = "high"
    def read_folder(self, folder):
        """Reads a folder of diamon output files, for varying file types

        Args:
            folder (str - path to file)

        Returns:
            dic: dict of df 
        """
        self.file_name = folder.split("\\")[-1]
        files_list = glob.glob(folder+"\*")
        for file in files_list:
            if "unfold" in file:
                self.read_unfold_file(file)
            elif "counters" in file:
                self.read_counters(file)
            elif "OUT_data" in file:
                self.out_data = read_data_file(file, 0, 3)
                self.out_data = self.out_data.drop(columns="INTERNAL")
                self.out_data["datetime"] = (
                    pd.to_timedelta(self.out_data['t(s)'], unit='s') + 
                    self.start_time)
            elif "rate" in file:
                self.rate_data = read_data_file(file, 0, 1)
            elif "LONG_OUT" in file and self.summary == True:
                # long out data gives the dose and %energy neutrons as func of time in set time intervals 
                self.summary_out = read_data_file(file, 0, 3)
            elif "LONG_rate" in file and self.summary == True:
                # long rate data gives all counts as func of time for each detector in set time intervals
                self.summary_rate = read_data_file(file, 0, 1)
            else:
                print("Error: please input a valid folder directory")
                break

    def get_shutter_info(self, shutter):
        beamlines = da.get_east_west_names(self)
    def filter_current(self, current_df, times, i):
        time = times[i]
        #first measurement take last data 20s before
        if i ==0:
            start = times[i] - np.timedelta64(20, 's')
        else:
            #take start as value before
            start = times[i-1]
        filtered_current = current_df.loc[pd.IndexSlice[:, start:time], :]["_value"]
        return np.mean(filtered_current)
    def shutter_filter(self, shutter_df, current_df):
        values = []
        current_values = []
        times = np.array(self.out_data["datetime"])
        for i, time in enumerate(times):
            shutter_df 
            values.append(shutter_df.loc[pd.IndexSlice[:, :time], :].groupby(shutter_df.index.names[0]).tail(1).droplevel("datetime"))
            current_values.append(self.filter_current(current_df, times, i))
        return values

def load_diamon(loc_data, fname, shutter_data):
    """data from diamon instrument loaded from files. data matched with measurement coordinate

    Args:
        loc_data (df): pandas dataframe of x, y, z coordinates and data reference
        fname (str): folder directory to diamon data

    Returns:
        dictionary of diamon class objects
    """
    diamon_list = []
    series_list = []
    for folder in glob.glob(fname):
        diamon_obj = diamon(False)
        diamon_obj.read_folder(folder)
        diamon_obj.folder_name = os.path.basename(folder)
        #print(diamon_obj.out_data)
        diamon_list.append(diamon_obj)
        series_list.append(diamon_obj.id)
    # match the location from the file name ref to coord to get shutter info
    # add option to read in shutter info relevent to obj
    id = da.get_measurement_id(pd.DataFrame(series_list))
    shutter_df = pd.concat(shutter_data["shutters"].values(), keys=shutter_data["shutters"].keys(), names=["beamline", "datetime"])
    current_df = pd.concat(shutter_data["current"].values(), keys=shutter_data["current"].keys(),  names=["target_station", "datetime"])
    for data in diamon_list:
        data = match_location(loc_data, data, id)
        data.beamlines = da.get_east_west_names(data.reference)
        data.beamline = da.get_names(data.reference["Measurement Reference"].iloc[0])[1]
        #data.filter_shutters(shutter_df, current_df)
        #shutter_info = [sa.filter_shutters([shutter_df, current_df], data.start_time, time, [shutter_df.index.names[0], current_df.index.names[0]]) for time in data.out_data["datetime"]]
    return diamon_list

def split_influx_data(influx_dic, splitby="current"):
    """Split shutter and current information into 2 keys of a nested dic

    Args:
        influx_dic (_type_): _description_
    """
    if splitby == "current":
        keys = ["ts1_current", "ts2_current"]
        

def match_location(location_data, data, id):
    """match location csv file key to data file name

    Args:
        location_data (df): pd df of xyz and ref.
        diamon_list (list): list of diamon class obj
        series_list (list): list of pd data series containing file id's

    Returns:
        dic: list of diamon obj with keys assigned
    """
    # load location
    data.file_name = id[id["start"] == data.start_time]["key"].values[0]
    data.reference = location_data.loc[location_data["Name"] == data.file_name].reset_index()
    return data

def read_data_file(path, i, j):
    """_summary_

    Args:
        path (_type_): _description_
        i (_type_): _description_
        j (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = pd.read_csv(path, sep='\t', index_col=False)
    data = data.dropna(axis='columns')
    data = data.drop(data.iloc[:, i:j], axis=1)
    data = data.replace('\%', '', regex=True)
    for col in data.columns:
        if '%' in col:
            data[col]= data[col].astype(float)
    return data

def get_measurement_id(df):
    """generates unique id based on serial and which repeat of the day

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    sorted_df = df.sort_values(by="start").reset_index()
    sorted_df["count"] = sorted_df.groupby(by=[sorted_df['start'].dt.date, sorted_df["serial"]]).cumcount()+1
    sorted_df["key"] = ((sorted_df["start"].dt.strftime("%d.%m.")) + (sorted_df["count"].astype(str)) + "-" + sorted_df["serial"])
    return sorted_df

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

def read_data(folder_path, location_path, shutter_path):
    
    shutter_data = na.load_pickle(shutter_path)
    shutter_data = sa.check_updated_shutter_info(shutter_data)
    loc_data = na.read_csv(location_path, {'x': float, 'y': float, 'z':float})
    diamon_data = load_diamon(loc_data, folder_path, shutter_data)
    return diamon_data
