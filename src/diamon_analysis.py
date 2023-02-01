import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import diamon_read_data as dia
import argparse
from scipy.signal import find_peaks
import influx_data_query as idb
import pickle

def load_data(data_path, location_path, building):

    data = dia.read_folders(data_path, location_path)
    if building == "ts1":
        ts1 = filter_location(data, "TS1")
        return ts1
    elif building == "ts2":
        ts2 = filter_location(data, "TS2")
        return ts2
    elif building == 'all':
        return data
    else:
        print("invalid selection: try again")
        return None
def save_pickle(data, name):
    
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def check_key_error(dic, key):
    if key in dic.keys():
        return dic

def filter_by_date(data, date):
    
    return [selected_data for selected_data in data if date in selected_data]

def filter_beamline(data, instrument):
    filtered_data = {key: get_beamline(measurement, instrument) for key, measurement in data.items() if get_beamline(measurement, instrument)}
    return filtered_data

def get_beamline(data, instrument):
    ref = data["reference"]["Measurement Reference"].iloc[0]
    #instead of returning the same data return the dataframe here?
    match instrument:
        case "chipir":
            series = {}
            return {key: dic for key, dic in data.items() if ref[1] == "C"}
        case "imat":
            return {key: dic for key, dic in data.items() if ref[1] == "I"}
        case "inter":
            return {key: dic for key, dic in data.items() if ref[1:3] == "In"}
        case "larmor":
            return {key: dic for key, dic in data.items() if ref[1] == "L"}
        case "let":
            return {key: dic for key, dic in data.items() if ref[1:3] == "Le"}
        case "nimrod":
            return {key: dic for key, dic in data.items() if ref[1] == "N"}
        case "offspec":
            return {key: dic for key, dic in data.items() if ref[1] == "O"}
        case "polref":
            return {key: dic for key, dic in data.items() if ref[1] == "P"}
        case "sans2d":
            return {key: dic for key, dic in data.items() if ref[1] == "S"}
        case "wish":
            return {key: dic for key, dic in data.items() if ref[1] == "W"}
        case "zoom":
            return {key: dic for key, dic in data.items() if ref[1] == "Z"}
        case "epb":
            return {key: dic for key, dic in data.items() if ref[1] == "B"}

def convert_to_df(data):
    """Filter out any low energy recordings for high energy unfold data
    returns a dataframe
    """
    beam_df = {}
    for beamline in beamline_names():
        series_list = []
        for measurement in data.values():
            ref = measurement["reference"]["Measurement Reference"].iloc[0]
            match beamline:
                case "chipir":
                    if ref[1] == "C":
                        series_list.append(dia.convert_to_ds(measurement))
                case "imat":
                    if ref[1] == "I":
                        series_list.append(dia.convert_to_ds(measurement))
                case "inter":
                    if ref[1:3] == "In":
                        series_list.append(dia.convert_to_ds(measurement))
                case "larmor":
                    if ref[1] == "L":
                        series_list.append(dia.convert_to_ds(measurement))
                case "let":
                    if ref[1:3] == "le":
                        series_list.append(dia.convert_to_ds(measurement))
                case "nimrod":
                    if ref[1] == "N":
                        series_list.append(dia.convert_to_ds(measurement))
                case "offspec":
                    if ref[1] == "O":
                        series_list.append(dia.convert_to_ds(measurement))
                case "polref":
                    if ref[1] == "P":
                        series_list.append(dia.convert_to_ds(measurement))
                case "sans2d":
                    if ref[1] == "S":
                        series_list.append(dia.convert_to_ds(measurement))
                case "wish":
                    if ref[1] == "W":
                        series_list.append(dia.convert_to_ds(measurement))
                case "zoom":
                    if ref[1] == "Z":
                        series_list.append(dia.convert_to_ds(measurement))
                case "epb":
                    if ref[1] == "B":
                        series_list.append(dia.convert_to_ds(measurement))
        beam_df[beamline] = pd.DataFrame(series_list)
    return beam_df

#add parallelisation?
def filter_location(data, building, beamline=None):
    match building:
        case "TS1":
            ts1_dict = {key: dic for key, dic in data.items() if (
                dic["reference"]["Measurement Reference"].iloc[0][0] == '1')} 
            return ts1_dict
        case "TS2":
            ts2_dict = {key: dic for key, dic in data.items() if (
                dic["reference"]["Measurement Reference"].iloc[0][0] == '2')}
            return ts2_dict

def get_shutter_name(data):
    ref = data["reference"]["Measurement Reference"].iloc[0]
    match ref[1]:
        case "N":
            return "t2shut::nimrod:status"
        case "L":
            if ref[2] == 'e':
                return "t2shut::let:status"
            else:
                return "t2shut::e6:status"
        case "I":
            if ref[2] == 'n':
                return "t2shut::inter:status"
            else:
                return "t2shut::w5:status"
        case "C":
            return "t2shut::chipir:status"
        case "Z":
            return "t2shut::e1:status"
        case "S":
            return "t2shut::sans2d:status"
        case "P":
            return "t2shut::polref:status"
        case "O":
            return "t2shut::offspec:status"
        case "W":
            return "t2shut::wish:status"

def beamline_names():
    return ["chipir", "sans2d", "wish", 
                "inter", "offspec", "let", 
                "nimrod", "polref", "zoom", "larmor", 
                "imat", "epb"]

def get_shutter_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def filter_position(data, coordinate: tuple):
    
    return

def convert_time_datetime(data):
    #convert data time table into datetime 
    for name in data.keys():
        if 'datetime' in data[name].keys():
            data[name]['out']['datetime'] = pd.to_timedelta(data[name]['out']['t(s)'], unit='s') + data[name]['datetime']['start']
    return data



def select_high_energy(data):

                
    return  {data_key: date for data_key, date in data.items() for key in date.keys()  if "high_e" in key}

def background_subtraction(data,background):
    
    return data - background

def remove_times(data, time_range):
    
    return 0

def average_daily_data(unfold_data):
    
    fluxes = []
    
    for data in unfold_data:
        
        _, flux = extract_spectrum(data)
        if np.average(flux) > 1e-10:
            fluxes.append(flux)
    avg_flux = np.average(fluxes, axis=0)
    return avg_flux

def find_std_dev(data):

    std = np.std(data)
    return std

def neuron_energy_dist():
    return 0

def extract_spectrum(data):
    energy = data["energy_bins"]
    flux = data["flux_bins"]
    return energy, flux

def get_energy_range(unfold_dataseries):
    
    
    return

def peaks_finder(data):
    
    energy, flux = np.array(extract_spectrum(data))
    #border threshold to reflect change in signal
    #border = np.
    
    flux_peaks_i , flux_peaks = find_peaks(flux, height=0, prominence=0.0001)
    flux_peaks = flux_peaks["peak_heights"]

    energy_peaks = energy[(flux_peaks_i)]
    return flux_peaks, energy_peaks
"""
def get_energy_range(unfold_data):
    """"""Returns a dict of thermal epi and fast energies""""""
    if isinstance(unfold_data, pd.DataFrame):
        energy = {}
        energy["thermal"] = unfold_data.thermal
        energy["epithermal"] = unfold_data.epi
        energy["fast"] = unfold_data.fast        
        
        return energy
"""
def dominant_energy(energy):
    dominant = max(energy, key=energy.get)
    return dominant

def find_significant_fast(energy):

    if max(energy, key=energy.get) == 'fast' and max(energy.values()) > 0.5:
        return energy
def fit_gaussian_spect():
    
    return 0

def find_abs_error(dataframe):
    
    for i, col in enumerate(dataframe.columns):
        if 'un%' in col:
            dataframe["abs_err " + dataframe.columns[i-1]] = dataframe[dataframe.columns[i-1]] * (dataframe[col]/100)
            
    return dataframe

def find_spect_peaks(data):
    
    energy_list = []
    flux_list = []
    for spectra in data:
        
        energy, flux = peaks_finder(spectra)
        energy_list.append(energy)
        flux_list.append(flux)
    return energy_list, flux_list

def get_unfold_dataframe(all_data):
    
    #convert to data series
    unfold_datalist = []
    for data in all_data:
        unfold_data = dia.convert_to_ds(data)
        unfold_datalist.append(unfold_data)
  
    unfold_dataframe = pd.DataFrame(unfold_datalist)
    return unfold_dataframe

def get_distance(data):
    col = ['x', 'y', 'z']
    df = data.loc[data[col].notnull().all(axis=1)]
    data['distance'] = np.sqrt(df['x']**2 + df['y']**2)
    return data

def convert_date_to_string(dt):
    return str(np.datetime_as_string(dt))

def get_current_status(shutters, df, i, shutter_name, start, end):
    
    start_status = shutters[shutter_name][shutters[shutter_name]["_time"].values < df['start'].values[i]].tail(1)
    end_status = shutters[shutter_name][shutters[shutter_name]['_time'].between(start, end)]
    status = start_status.merge(end_status, how='outer')
    return status

def get_shutter_status(shutter, df, i, start, end):
    status_start = shutter[shutter['_time'].values < df['start'].values[i]].tail(1)
    status_end = shutter[shutter['_time'].between(start, end)]
    return status_start.merge(status_end, how='outer')

def filter_shutter_status(beam_df, shutters):
#This function extracts the shutter and beam current status at time of measurement.
    shutter_dict = {}
    for name, df in beam_df.items():
        if not df.empty:
#------------------------------------------------------------------------------------------#
            for i, filename in enumerate(df['file_name']):
                start = convert_date_to_string(df['start'].values[i])
                end = convert_date_to_string(df['end'].values[i])
                if name == "epb":
                    shutter_dict[str(filename)] = {}
                    shutter_dict[str(filename)]["ts1_curr"] = get_current_status(shutters, df, i, "ts1_current", start, end)
                    shutter_dict[str(filename)]["ts2_curr"] = get_current_status(shutters, df, i, "ts2_current", start, end)
                else:
                    shutter = shutters[name]

                    shutter_dict[str(filename)] = {}
                    shutter_dict[str(filename)]["shutter_status"] = get_shutter_status(shutter, df, i, start, end)
                    shutter_dict[str(filename)]["ts1_curr"] = get_current_status(shutters, df, i, "ts1_current", start, end)
                    shutter_dict[str(filename)]["ts2_curr"] = get_current_status(shutters, df, i, "ts2_current", start, end)
    return shutter_dict

def mean_curr_between_date(df, current_df):
    currents = []
    times = df["out"]["datetime"]
    for i, _ in enumerate(times):
        if i == 0:
            start = convert_date_to_string(current_df["_time"].values[i])
            end = convert_date_to_string(times.values[i])
        else:
            start = convert_date_to_string(times.values[i-1])
            end = convert_date_to_string(times.values[i])
        current = np.mean(current_df["_value"][current_df["_time"].between(start, end)])
        currents.append(current)
    return currents
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diamon Analysis")
    parser.add_argument('file_name', type=argparse.FileType('r'), help="Enter the filename")
    args = parser.parse_args()
    dia.read_folder(args.file_name)"""
    