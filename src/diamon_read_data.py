# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:55:42 2022

This file will process both the F_UNFOLD and the rate data from DIAMON spectrometer
"""
import numpy as np
import pandas as pd
import glob
import pickle
import re
import diamon_analysis as da
from datetime import datetime, timezone, timedelta
from pathlib import Path
from memory_profiler import profile
import influx_data_query as idb
import dask.dataframe as dd

def load_pickle(name):
    "unpickles a file and loads into the script"
    with open(name, 'rb') as f:
        return pickle.load(f)

def read_diamon_folders(data_path, location_path, shutter_path):
    """
    Reads diamon folders, location data and loads in shutter information 
    from idb query
    """
    folder_list = glob.glob(data_path)
    shutters = load_pickle(shutter_path)
    # ensures location coordinates read in as float not str
    locations = pd.read_csv(location_path,  dtype={'x': float, 'y': float, 'z':float})
    all_data = {}
    # loop over diamon folders reading each set of files and filtering shutters
    for folder in folder_list:
        data = read_folder(folder, locations)
        #data = da.filter_shutters(data, shutters)
        #data = da.normalise_dose(data)
        #data["summary"] = da.convert_to_ds(data)
        all_data[data["name"]] = data
    return all_data

def read_folder(folder, locations):
    """Reads a folder of diamon output files, for varying file types

    Args:
        folder (str - path to file)

    Returns:
        dic: dict of df 
    """
    files_list = glob.glob(folder+"\*")
    dic = {}
    dic["name"] = folder.split("\\")[-1]
    dic["reference"] = locations.loc[locations["Name"] == dic["name"], 
                                     locations.columns != "index"]
    for file in files_list:
        if "C_unfold" in file:
            c_unfold = read_unfold_file(file)
            c_unfold["energy_mode "] =  "high"
            dic["high_e"] = c_unfold
        elif "counters" in file:
            counters = read_counters(file)
            dic["datetime"] = counters
        elif "F_unfold" in file:
            f_unfold = read_unfold_file(file)
            f_unfold["energy_mode"] =  "low"
            dic["low_e"] = f_unfold
        elif "LONG_OUT" in file:
            out_data = read_data_file(file, 0, 3)
            dic["long_out"] = out_data
        elif "LONG_rate" in file:
            rate_data = read_data_file(file, 0, 1)
            dic["long_rate"] = rate_data
        elif "OUT_data" in file:
            #out data gives the dose and %energy neutrons as func of time in set time intervals (6 measurements)
            out_data = read_data_file(file, 0, 3)
            dic["out"] = out_data
        elif "rate" in file:
            #rate data gives all counts as func of time for each detector in set time intervals (6 measurements)
            rate_data = read_data_file(file, 0, 1)
            dic["rate"] = rate_data
        else:
            print("Error: please input a valid folder directory")
            break
    if "datetime" in dic.keys():
        dic["out"]["datetime"] = pd.to_timedelta(dic['out']['t(s)'], unit='s') + dic['datetime']['start']
        #dic["out"]["datetime"] = dic["out"]["datetime"].apply(lambda d: d.replace(tzinfo=timezone.utc))
    else:
        print("Error in diamon folder: " + dic["name"] + " - no counters file - check data")
        exit()
    return dic

def read_unfold_file(path):
    """Read diamon c/f unfold extracting energy distributions

    Args:
        path (_type_): _description_
    Returns:
        _type_: _description_
    """
    in_spect = False
    dic = {}
    dic["file_name"] = Path(path).stem
    energy_bins = []
    flux_bins = []
    with open(path) as f:
        for line in f:
            if " thermal" in line:
                dic["thermal"] = clean_param(line)
            elif "epi" in line:
                dic["epi"] = clean_param(line)
            elif "fast" in line:
                dic["fast"] = clean_param(line)
            elif "phi" in line:
                dic["phi"], dic["phi_uncert"] = clean_param(line, True)
            elif "H*(10)_r" in line:
                dic["dose_rate"],  dic["dose_rate_uncert"] = clean_param(line, True)
            elif "h*(10)" in line:
                dic["dose_area_product"], dic["dose_area_product_uncert"] = clean_param(line, True)
            elif "D1" in line:
                dic["count_D1"], dic["count_R"] = clean_counts(line)
            elif "D2" in line:
                dic["count_D2"], dic["count_RL"] = clean_counts(line)
            elif "D3" in line:
                dic["count_D3"], dic["count_FL"] = clean_counts(line)
            elif "D4" in line:
                dic["count_D4"], dic["count_F"] = clean_counts(line)
            elif "D5" in line:
                dic["count_D5"], dic["count_FR"] = clean_counts(line)
            elif "D6" in line:
                dic["count_D6"], dic["count_RR"] = clean_counts(line)
            elif "TIME" in line:
                dic["time"] = float(clean_param(line))
            elif in_spect and ("----" not in line):
                line = clean(line)
                if len(line)<1:
                    break
                energy_bins.append(float(line[0]))
                flux_bins.append(float(line[-1]))
            elif "Ec" and "Phi(E)*Ec" in line:
                in_spect = True
    f.close()
    dic["energy_bins"] = energy_bins
    dic["flux_bins"] = flux_bins
    return dic

def read_counters(path):
    with open(path) as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if "Measurement Time" in lines[i-1]:
                time = " ".join(line.strip().split())
            elif "GMT time" in lines[i-1]:
                start_time = datetime.strptime(line, "%a %b %d %H:%M:%S %Y")
                start_time = start_time.replace(tzinfo=timezone.utc)
                end_time = start_time + timedelta(seconds=float(time))
    f.close()
    return {"start": start_time, "end": end_time}

def read_data_file(path, i, j):
    data = pd.read_csv(path, sep='\t', index_col=False)
    data = data.dropna(axis='columns')
    data = data.drop(data.iloc[:, i:j], axis=1)
    data = data.replace('\%', '', regex=True)
    for col in data.columns:
        if 'un%' in col:
            data[col]= data[col].astype(float)
    return data

#set of functions for string cleaning
def clean_param(line, uncert=None):
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

if __name__ == '__main__':
    path = r"C:\Users\sfs81547\OneDrive - Science and Technology Facilities Council\Documents\ISIS\Diamon Project\TS2 Measurements\DIAMON*"
    folders = glob.glob(path)
    d = read_diamon_folders(path)
