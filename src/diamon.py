import pandas as pd
from datetime import datetime, timedelta
import glob
import re
import pytz
import src.neutronics_analysis as na
import math
import src.beamline as sa

loc_path = r"data\measurement_location.csv"
target_station_path = r"data\target_station_data.csv"
diamon_path = r"data\measurements\DIAMON*"

class diamon:
    """
    This class creates an instance from a diamon measurement. The class has the ability
    to load results from the diamon detector, sort and filter them into different attributes
    of the object. Requires a path to diamon measurement folder and location spreadsheet to match id and 
    date to a coordinate system.
    """
    def __init__(self, folder_path=""):
        self.out_data : pd.DataFrame
        self.rate_data : pd.DataFrame
        self.datetime : datetime
        self.file_name = ""
        self.high_energy_bin = []
        self.high_flux_bin = []
        self.dose : float
        self.unfold_data = pd.Series(dtype=object)
        self.energy_type = "low"
        self.time = None
        self.reference = None
        self.start_time = None
        self.end_time = None
        self.id = None
        self.current_data = []
        if folder_path != "":
            self.read_folder(folder_path)
            self.folder_name = folder_path.split("\\")[-1]

    def __str__(self):
        return self.file_name

    def read_counters(self, path :str):
        """
        This function reads DIAMON "counters.txt" file and loads in the
        start, end time and id of diamon.
        Args:
            path (str): path to counters file
        """
        with open(path) as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                if "Measurement Time" in lines[i-1]:
                    # how long diamon was recording for
                    self.time = " ".join(line.strip().split())
                elif "GMT time" in lines[i-1]:
                    start_time = datetime.strptime(line, "%a %b %d %H:%M:%S %Y")
                    self.start_time = pytz.timezone("Europe/London").localize(start_time)
                    self.end_time = self.start_time + timedelta(seconds=float(self.time))
                elif "Serial" in line:
                    # which serial number of diamon is used
                    serial = re.findall(r"\d$", line)[0]
        f.close()
        self.id = pd.Series({"start":self.start_time, "serial":serial})
 
    def read_unfold_file(self, path :str):
        """Read diamon c/f unfold extracting energy distributions

        Args:
            path (str): path to unfold file
        """
        self.unfold_data = {}
        spect_data = False
        energy_bin = []
        flux_bin = []
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
                    self.unfold_data["time"] = float(na.clean_param(line))
                elif spect_data and ("----" not in line):
                    line = na.clean(line)
                    if len(line) < 1:
                        break
                    energy_bin.append(float(line[0]))
                    flux_bin.append(float(line[-1]))
                elif "Ec" and "Phi(E)*Ec" in line:
                    spect_data = True
        f.close()
        # if energy is high or low energy
        if re.findall(r"[^\/]*C_unfold[^\/]*$", path):
            self.energy_type = "high"
            self.high_energy_bin = energy_bin
            self.high_flux_bin = flux_bin
        elif re.findall(r"[^\/]*F_unfold[^\/]*$", path):
            self.low_energy_bin = energy_bin
            self.low_flux_bin = flux_bin

    def read_folder(self, folder):
        """Reads a folder of diamon output files, for varying file types and stores
        inside the diamon class

        Args:
            folder (str - path to file)
        """
        self.file_name = folder.split("\\")[-1]
        files_list = glob.glob(folder+"\*")
        if files_list == []:
            raise Exception("no directory found please check the name entered")
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
                raise Exception("Error: Unknown file in directory: " + str(folder))

    def get_shutter_name(self):
        """gets shutter name of instrument the diamon was placed at

        Returns:
            _type_: _description_
        """
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

def read_data_file(path : str, i : int, j : int):
    """
    read a data file from the detector (either rate/out)
    Args:
        path (str): path to file
        i (int): which data row to slice
        j (int): which data column to slice

    Returns:
        pd.DataFrame: returns panadas dataframe of data from file
    """
    data = pd.read_csv(path, sep='\t', index_col=False)
    data = data.dropna(axis='columns')
    data = data.drop(data.iloc[:, i:j], axis=1)
    data = data.replace('\%', '', regex=True)
    for col in data.columns:
        if '%' in col:
            data[col]= data[col].astype(float)
    return data

def clean_counts(line):
    # specific cleaning function for line
    line = na.clean(line)
    return int(line[1]), int(line[3])

def set_beamline_info(data):
    """
    create diamon attribute defining the reference and creating an instance
    of the beamline class
    Args:
        data (dict[diamon]): dict of all diamon data
    Returns:
        dict: dict containing diamon data with references matched
    """
    location_data = pd.read_csv(loc_path, dtype={'x': float, 'y': float, 'z':float})
    beamline_df = pd.read_csv(target_station_path, index_col=["Building", "Location"])
    for result in data.values():
        ref = result.reference["Measurement Reference"].iloc[0]
        if ("BL" not in ref) or ("BB" not in ref) or ("BT" not in ref):
            result.reference = location_data.loc[location_data["Name"] == result.file_name].reset_index()
            result.beamlines = sa.beamline(ref, beamline_df)
    return data
