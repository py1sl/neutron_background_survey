import pandas as pd
from datetime import datetime, timedelta
import glob
from pathlib import Path
import re
import pytz
import src.diamon_analysis as da
import src.neutronics_analysis as na
import math
import src.shutter_analysis as sa

loc_path = "data\measurement_location.csv"
diamon_path = "data\measurements\DIAMON*"

class diamon:
    def __init__(self, folder_path="", summary_out_selected=False):
        
        self.out_data : pd.DataFrame()
        self.rate_data : pd.DataFrame()
        self.datetime : datetime
        self.file_name = ""
        self.energy_bin = []
        self.flux_bin = []
        self.dose : float
        self.unfold_data = pd.Series(dtype=object)
        self.shutters = []
        self.energy_type = "low"
        self.time = None
        self.start_time = None
        self.end_time = None
        self.id = None
        self.summary = summary_out_selected
        self.current_data = []
        if folder_path != "":
            self.read_folder(folder_path)
            self.folder_name = folder_path.split("\\")[-1]

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
                    self.end_time = self.start_time + timedelta(seconds=float(self.time))
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
        self.unfold_data  ={}
        spect_data = False
        with open(path) as f:
            for line in f:
                if " thermal" in line:
                    self.unfold_data["thermal"] = na.clean_param(line)
                elif "epi" in line:
                    self.unfold_data["epi"] = na.clean_param(line)
                elif "fast" in line:
                    self.unfold_data["fast"] = na.clean_param(line)
                elif "phi" in line:
                    self.unfold_data["phi"], self.unfold_data["phi_uncert"] = na.clean_param(line, True)
                elif "H*(10)_r" in line:
                    self.unfold_data["dose_rate"],  self.unfold_data["dose_rate_uncert"] = na.clean_param(line, True)
                elif "h*(10)" in line:
                    self.unfold_data["dose_area_product"], self.unfold_data["dose_area_product_uncert"] = na.clean_param(line, True)
                elif "D1" in line:
                    self.unfold_data["count_D1"], self.unfold_data["count_R"] = na.clean_counts(line)
                elif "D2" in line:
                    self.unfold_data["count_D2"], self.unfold_data["count_RL"] = na.clean_counts(line)
                elif "D3" in line:
                    self.unfold_data["count_D3"], self.unfold_data["count_FL"] = na.clean_counts(line)
                elif "D4" in line:
                    self.unfold_data["count_D4"], self.unfold_data["count_F"] = na.clean_counts(line)
                elif "D5" in line:
                    self.unfold_data["count_D5"], self.unfold_data["count_FR"] = na.clean_counts(line)
                elif "D6" in line:
                    self.unfold_data["count_D6"], self.unfold_data["count_RR"] = na.clean_counts(line)
                elif "TIME" in line:
                    self.unfold_data["time"] = float(na.clean_param(line))
                elif spect_data and ("----" not in line):
                    line = na.clean(line)
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
    @property
    def beam_name(self):
        return self.beamlines.name

    def find_distance(self, dimension=2):
        """
        get 2d and 3d pythag distance between coordinates and the origin
        Args:
            self (diamon class)
            dimension (int, optional): 2d or 3d dimension. Defaults to 2.
        """
        self.x = self.reference["x"].iloc[0]
        self.y = self.reference["y"].iloc[0]
        if dimension == 2:
            self.distance =  math.sqrt(self.x**2 + self.y**2)
        elif dimension == 3:
            self.z = self.reference["z"].iloc[0]
            self.distance = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        else:
            raise Exception("invalid dimension - only 2 or 3 allowed")

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

# read all data
def read_data(shutter_data):
    # try to load exisiting data

    data = da.load_pickle("diamon_data")
    loc_data = na.read_csv(loc_path, {'x': float, 'y': float, 'z':float})
    if data:
        print("checking if there is new data in the directory")
        data = update_diamon_data(data, loc_data, shutter_data)
    else:
        print("no data - loading all data in file path")
        data = load_diamon(loc_data, shutter_data)
    return data

def update_diamon_data(existing_data, loc_data, shutter_data):
    """_summary_

    Args:
        existing_data (_type_): _description_
        loc_data (_type_): _description_
        shutter_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # check if missing
    latest_data = get_recent_entry(existing_data)
    file_paths = glob.glob(r"data\ts2_measurements\DIAMON*\counters.txt")
    unread_paths = [x for file_path in file_paths if (x := read_unloaded_file(file_path, latest_data)) is not None]
    if len(unread_paths) == 0:
        print("No new data - loading existing file")
        return existing_data
    else:
         new_dic = load_diamon(loc_data, unread_paths, shutter_data)
         existing_data.update(new_dic)
         # SAVE NEW DATA
         na.save_pickle(existing_data, "main_data")
         return new_dic

def get_recent_entry(data):
    # in exisitng group fo diamon results get most recent entry
    return max([file.start_time for file in data.values()])

def read_unloaded_file(file_path, latest):
    match_string = "GMT time"
    # record names of files to read in
    lines = na.read_lines(file_path)
    for i, line in enumerate(lines):
        if match_string in lines[i-1]:
            aware_str = na.str_to_local_time(line)
            if aware_str > latest:
                return Path(file_path).parent

# update info
def load_diamon(loc_data, shutter_data):
    """data from diamon instrument loaded from files. data matched with measurement coordinate
    Args:
        loc_data (df): pandas dataframe of x, y, z coordinates and data reference
        fname (str): folder directory to diamon data

    Returns:
        dictionary of diamon class objects
    """
    diamon_list = [diamon(folder) for folder in glob.glob(diamon_path)]
    # match the location from the file name ref to coord to get shutter info
    diamon_list = match_id_location(loc_data, diamon_list)
    diamon_dict =  {data.file_name: data for data in diamon_list}
    return diamon_dict

def match_id_location(loc_data, data):
    id_list = [diamon_obj.id for diamon_obj in data]
    id = get_measurement_id(pd.DataFrame(id_list))
    data = [match_location(loc_data, diamon_obj, id) for diamon_obj in data]
    return data

def get_measurement_id(df):
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
    # NEED MOVE THIS
    beamline_df = pd.read_csv("data/target_station_data.csv", index_col=["Building", "Location"])
    data.beamlines = sa.beamline(data.reference["Measurement Reference"].iloc[0], beamline_df)
    return data

def set_beamline_info(data):
    location_data = pd.read_csv("data/measurement_location.csv", dtype={'x': float, 'y': float, 'z':float})
    beamline_df = pd.read_csv("data/target_station_data.csv", index_col=["Building", "Location"])
    for result in data.values():
        result.reference = location_data.loc[location_data["Name"] == result.file_name].reset_index()
        result.beamlines = sa.beamline(result.reference["Measurement Reference"].iloc[0], beamline_df)
