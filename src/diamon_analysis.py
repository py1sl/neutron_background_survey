import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import pickle
from datetime import datetime
import dask.dataframe as dd
from collections import defaultdict
import src.influx_data_query as idb
from src.diamon import diamon

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
    try:
        with open(name, 'rb') as f:
            return pickle.load(f)
    except IOError:
        print("no file exists")
        return None
def influx_db_query(dates, names=None):
    """"
    load influx database between selected dates and option include specific channel names
    args:
        dates (list of datetime as STR)
        names (optiponal): df of beam info and channel names
    """
    query_obj = idb.query_object(names, True)
    query_obj.start = dates[0]
    query_obj.end = dates[1]
    query_data = query_obj.influx_query(names, update=True)
    return query_data


def filter_location(data, building):
    """
    filter data if both ts1 & ts2 read in into separate dictionaries based on reference
    Args:
        data (dict): all ts1 ts2 data
        building (string): selected building to filter

    Returns:
        _type_: _description_
    """
    #match the variable building to the string - building
    if building == "TS1":
            ts1_dict = {key: dic for key, dic in data.items() if (dic.beamlines.target_station == '1')} 
            return ts1_dict
    elif building ==  "TS2":
            ts2_dict = {key: dic for key, dic in data.items() if (dic.beamlines.target_station == '2')}
            return ts2_dict

def convert_to_ds(data, labels):
    """
    Convert list of labels and the data into a pandas series
    Args:
        dic (dict) : all data for one measurement

    Returns:
        dseries
    """
    labels = ("name", "start", "end", "reference", "x", "y", "z", "dose_rate", "norm_dose",
              "dose_rate_uncert")

    # the unfold data - summary of diamon data in high or low energy mode 
    # filter for high or low energy

    unfold_data = data.unfold_data

    data_list = [data.file_name, data.start_time, data.end_time, 
                 data.reference["Measurement Reference"].iloc[0], 
                 data.reference['x'].iloc[0], data.reference['y'].iloc[0], 
                 data.reference['z'].iloc[0], unfold_data.dose_rate, data.out_data["norm_dose"].iloc[-1],
                 unfold_data.dose_rate_uncert]

    s1 = pd.Series(data_list, index=labels)
    # call function to find 2d distance
    try:
        s1["distance"] = data.distance
    except AttributeError:
        # need get distance
        data.distance = data.find_distance()
    return s1

def convert_row_series(result, df):
    # converts a data series into a data frame and joins to existing df
    dseries = convert_status_to_ds(result)
    merged_df = pd.merge(dseries.to_frame().T, df, how="cross")
    return merged_df

def convert_status_to_ds(data):
    """
    basic data info put into a panda series (name, coordinates, location)
    Args:
        dic (dict): dictionary of all diamon data

    Returns:
        s1 - panda series: series containing key info
    """
    labels = ("key", "start", "end", "serial", "reference", "x", "y", "z")

    data_list = [data.file_name, data.start_time,  data.end_time, data.id.values[1],
                 data.reference["Measurement Reference"].iloc[0],
                 data.reference['x'].iloc[0], data.reference['y'].iloc[0],
                 data.reference['z'].iloc[0]]

    s1 = pd.Series(data_list, index=labels)
    # first get distance
    try:
        data.find_distance()
        s1["distance"] = data.distance
    except AttributeError:
        # need get distance
        data.distance = data.find_distance()
    return s1

def combine_continuous_data_files(dataframes, cum_time=None):
    """
    This function allows repeated measurements in the same location to be conjoined
    into one datafame

    Args:
        dataframes (panda df):
        cum_time (boolean, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    combined_dataframe = []
    for i, dataframe in enumerate(dataframes):

        if cum_time and i != 0:
            last_index = dataframes[i-1].iloc[-1,0]
            #this aligns the new files time with the previous so they are adjacent
            dataframe.iloc[:,0] = last_index + dataframe.iloc[:,0]

        combined_dataframe.append(dataframe)
    combined_dataframe = pd.concat(combined_dataframe, ignore_index=True)

    return combined_dataframe

def select_high_energy(data):
    """
    Select high energy data from files - ignore F_unfold - low diamon energy mode
    Args:
        data (dict): diamon data

    Returns:
        dict: returns data with only high energy data included
    """
    return  {data_key: date for data_key, date in data.items() for key in date.keys()  if "high_e" in key}

def average_daily_data(unfold_data):
    """
    extracts the average flux across group of measurements
    Args:
        unfold_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    fluxes = []
    for data in unfold_data:
        _, flux = extract_spectrum(data)
        if np.average(flux) > 1e-10:
            fluxes.append(flux)
    avg_flux = np.average(fluxes, axis=0)
    return avg_flux

def extract_spectrum(data):
    """
`   get the flux and enegry bins from unfold data
    Args:
        data (series/df): contains unfold file data

    Returns:
        2 arrays for flux and energy
    """
    if data.energy_type == "high":
        energy = data.high_energy_bin
        flux = data.high_flux_bin
    else:
        energy = data.low_energy_bin
        flux = data.low_flux_bin
    return energy, flux

def peaks_finder(data):
    """
    find the peak of spectrum and extract energy and flux
    Args:
        data (series)_
    Returns:
        2 arrays of peak flux and energy for three regions
    """

    energy, flux = np.array(extract_spectrum(data))
    #border threshold to reflect change in signal
    
    flux_peaks_i , flux_peaks = find_peaks(flux, height=0, prominence=0.0001)
    flux_peaks = flux_peaks["peak_heights"]

    energy_peaks = energy[(flux_peaks_i)]
    return flux_peaks, energy_peaks

def convert_date_to_string(dt):
    """
    convert a numpy datetime64 object to string (str is necessary) as np returns a np.str class
    Args:
        dt (numpy datetime object)
    Returns:
        str
    """
    return str(np.datetime_as_string(dt))


def get_current_info(data, current_df):
    """
    extract current information from time of data
    Args:
        data (dict): diamon data
        current (df): df of current data

    Returns:
        list: list of current data
    """
    times = np.array(data.out_data["datetime"])
    currents = []
    #loop over every timestamp in diamon recording
    for i, time in enumerate(times):
        time = time.tz_convert("Europe/London")
        #first measurement take last data 20s before
        if i ==0:
            start = times[i] - np.timedelta64(20, 's')
        else:
            #take start as value before
            start = times[i-1]
        #filter in between start and current time
        filtered_current = current_df.loc[start:time]["_value"]
        currents.append(np.mean(filtered_current))
    return  currents

def dominant_energy(energy):
    """
    find dominant energy 
    Args:
        energy (array): 3 energy values for thermal, epithermal and fast neutrons

    Returns:
        float: value at highest %
    """
    dominant = max(energy, key=energy.get)
    return dominant

def find_significant(energy, energy_type):
    """
    get dominant energy and only if is bigger than 50% of all neutrons
    Args:
        energy (array): _description_
        energy_type (str): thermal, epi or fast

    Returns:
        float:  max energy
    """
    if max(energy, key=energy.get) == energy_type and max(energy.values()) > 0.5:
        return energy

def find_abs_error(dataframe, un_col_name, col_name):
    """
    convert % error to absolute error on selected column names
    Args:
        dataframe (pandas df)

    Returns:
        df: abs error included in df out
    """
    dataframe["abs_error"] = dataframe[col_name] * (dataframe[un_col_name] / 100)
    return dataframe

def filter_shutter_status(data, selected_shutter="all"):
    """
    for each result get a df
    filter the dataframe by open or closed shutter
    """
    filtered_list = []
    for result in data.values():
        new_df = []
        if result.out_data.empty:
            continue
        #remove epb measurements with no shutter status
        # check has a valid shutter
        if "shutter-open" not in result.out_data.columns:
            #print("This result is not on a beamline")
            last_row = result.out_data.iloc[[-1]]
            new_df = convert_row_series(result, last_row).set_index("key")
            filtered_list.append(new_df)
            continue
        else:
            if selected_shutter == "closest":
                neighbours = result.beamlines.closest_neighbours.index.to_numpy()
                neighbours = np.append(neighbours, result.beamlines.name)
            else:
                neighbours = result.beamlines.all_neighbours.index.to_numpy()
            for neighbour in neighbours:
                change_df = last_row_shutter_change(result, neighbour)
                if change_df is not None:
                    new_df.append(change_df)
        # append last entry
        last_row = result.out_data.iloc[[-1]]
        combined_row = convert_row_series(result, last_row)
        new_df.append(combined_row)
        if selected_shutter == "closest":
            combined_df = pd.concat(new_df).set_index("key")
            new_df = check_multiple_shutters(neighbours, combined_df)
        else:
            new_df = pd.concat(new_df).set_index("key")
        filtered_list.append(new_df)
    df = pd.concat(filtered_list)
    return df

def last_row_shutter_change(result, shutter_name):
    """
    goes through shutter df and look for time of last change in shutter
    Args:
        result (dict): dictionary of result data

    Returns:
        list: list of shutter status
    """
    data = result.out_data
    # ignore transient data (< 1000 seconds)
    data = data[(result.out_data["t(s)"] > 1000)]
    #get df of data when shutter changes
    try:
        change_times = shutter_change(data, shutter_name)
    except KeyError:
        change_times = None
    if change_times is None:
        return
    change_times["shift"] = change_times["t(s)"].shift(1)
    change_times["shift"].fillna(0, inplace=True)
    change_times = change_times[(change_times["t(s)"] - change_times["shift"]) > 500]
    merged_df = convert_row_series(result, change_times)
    return merged_df

def check_multiple_shutters(names, df):
    # this function creates a bool based on multiple name conditions
    # function returns true if all are true but false if all are false and nan if neither
    shutter_keep = df.loc[:, names]
    df["truth"] = np.where(shutter_keep.all(axis=1), True, np.where((~shutter_keep).all(axis=1), False, np.nan))
    return df

def flag_shutter(data, shutter, flag=True):
    """
    boolean mask of data to get data matching selected flag. flag is true/false for open/closed
    """
    if shutter == "own":
        shutter = "shutter-open"
    data = data[(data[shutter] == flag) | (data[shutter].isna())]
    averaged_data = average_repeated_data(data).dropna(subset = ['x', 'y', 'norm_dose']).reset_index()
    return averaged_data

def average_repeated_data(df):
    """
    When measurement has multiple data for same date and location take average of data
    Args:
        df (dataframe): key information for data in df
    returns: filtered df with averages taken for repeats
    """
    keep = df[[ "reference", "start", "end", "shutter-open"]].groupby(df.index).first()
    df = df.infer_objects()
    filtered_df = df.groupby(["key"]).mean(numeric_only=True)
    averaged_df = pd.concat([filtered_df, keep], axis=1)
    return averaged_df

def filter_low_beam_current(data, minimum_current):
    """
    This function replaces any data in out iles where beam current less than the argument minimum current
    Args:
        data (dict): dict of all data information
        minimum_current (float): minimum current to include

    Returns:
        dict: same data with data at a time with current < minimum removed
    """
    data.out_data= data.out_data[data.out_data["ts2_current"] > minimum_current]
    return data

def find_repeats(data):
    ref = list(set(data["reference"].values))
    filtered_dict = {}
    for text in ref:
        joined_str = split_reference(text)
        if joined_str[1] != '':
            filtered_dict[joined_str[0]] = (data[data["reference"].str.match(joined_str[0])]).drop_duplicates()
    return filtered_dict

def localise_tz(data, timezone):
    # ensure non datetime entries are valid
    data["_time"] = dd.to_datetime(data["_time"], errors='coerce')
    data["_time"] = data["_time"].dt.tz_localize(timezone)
    return data

def filter_shutter_open(df, shutter_names):
    """
    filters df by selected shutter status
    Args:
        df (panda df): summary df of all diamon data
        shutter_names (list of str): shutter names

    Returns:
        _type_: _description_
    """
    filtered_df = df[ (df[shutter_names].all(axis=1)) & ~(df[shutter_names].isna())]
    return filtered_df


def filter_shutter_closed(df, shutter_names):
    """
    filters df by selected shutter status
    Args:
        df (panda df): summary df of all diamon data
        shutter_names (list of str): shutter names

    Returns:
        _type_: _description_
    """
    filtered_df = df[ (~(df[shutter_names]).any(axis=1)) & ~(df[shutter_names[0]].isna())]
    return filtered_df

def shutter_change(df: pd.DataFrame, shutter : str):
    """
    finds row in df when shutter parameter changes and outputs all rows which match
    Args:
        df (pd.DataFrame): 
        shutter (str): 

    Returns:
        change_times (pd.DataFrame): df of rows with change in shutter
    """
    filter = (df[shutter].ne(df[shutter].shift(-1)))
    change_times = df[filter].iloc[:-1]
    if change_times.empty:
        return None
    else:
        return change_times

def select_shutters(data : diamon, selected_shutters : str = "all", shutters : list[str]=None):
    """
    This function selects the beamlines according to chosen parameter
    Args:
        data (diamon):  diamon class instance
        selected_shutters (str): option of shuter
        shutters : list[str] : selected shutters to look at
    Returns:
        3 possible outcomes - "all" - every beamlline on that side is selected, 
                              "closest" - closest neighbours
                              " custom" - own selection
        
    """
    if selected_shutters == "all":
        names = data.beamlines.all_neighbours.index.to_numpy()
        sel_names = [name for name in names if name in data.out_data.columns]
    elif selected_shutters == "closest":
        names = data.beamlines.closest_neighbours.index
        # add own shutter
        names = np.append(names, data.beamlines.name)
        sel_names = [name for name in names if name in data.out_data.columns]
    elif selected_shutters == "custom":
        # check the selected shutter is valid
        sel_names = [shutter for shutter in shutters if shutter in data.beamlines.all_neighbours.index]
    return sel_names

def split_reference(text):
    #splits reference to location key separated by two hyphens
    groups = text.split('-')
    joined_str = '-'.join(groups[:2]), '-'.join(groups[2:])
    return joined_str

def find_repeats(df, data):
    #check for data with matching keys aka a repeat measurement and return all repeats
    ref = list(set(df["reference"].values))
    filtered_df_dic = defaultdict(dict)
    keys = []
    for text in ref:
        joined_str = split_reference(text)
        if joined_str[1] != '':
            keys.append(joined_str[0])
            #filtered_df_dic[joined_str[0]] = (df[df["reference"].str.match(joined_str[0])]).drop_duplicates()
    keys = list(set(keys))
    for result in data.values():
        ref = split_reference(result.reference["Measurement Reference"][0])[0]
        if ref in keys:
            filtered_df_dic[ref][result.file_name] = result
    return filtered_df_dic

def summary_df(df : pd.DataFrame, columns : list, save="", duplicates=True):
    # store a summary df of information to csv file
    df["real_time"] = pd.to_datetime(df["start"]).add(pd.to_timedelta(df["t(s)"], unit="s"))
    summary_df = df[columns]
    if duplicates == False:
        summary_df = summary_df.drop_duplicates("reference")
    if save != "":
        summary_df.to_csv("data/" + str(save) + ".csv")
    return summary_df

def split_df_axis(df, selected_z):
    """
    This function gets positive and negative x,y and selected z values
    """
    df.loc[:,"z"] = df.loc[:, selected_z]
    pos_df = df[df["y"] > 0][["x", "y", "z"]]
    neg_df = df[df["y"] < 0][["x", "y", "z"]]
    if pos_df.empty:
        return [neg_df]
    elif neg_df.empty:
        return [pos_df]
    elif pos_df.empty and neg_df.empty:
        raise Exception("empty df - check your df")
    else:
        return [pos_df, neg_df]

def compare_df(df, labels, status=[True, False], shutters=None):
    """
    filter df for selected flagged shutter
    """
    if shutters is None:
        df1 = flag_shutter(df, "own", status[0])
        df2 = flag_shutter(df, "own", status[1])
    else:
        df1 = flag_shutter(df, shutters[0], status[0])
        df2 = flag_shutter(df, shutters[1], status[1])
    df_dict = {labels[0]: df1, labels[1]: df2}
    return df_dict

def filter_reference_location(diamon_dict : dict[classmethod], condition : str, 
                              match_case : bool=False):
    """matches the reference key of a measurement to condition
    optional arg to set to exact or contains key

    Args:
        diamon_obj (dict[diamon]): diamon class instance in dictionary
        condition (str): string conditon eg: "2Im"
        match_case (str) optional : match exact ref or if contains ref in key
    """
    filtered_dict = {}
    for key, result in diamon_dict.items():
        if match_case == True:
            if result.reference["Measurement Reference"].iloc[0] == condition:
                filtered_dict[key] = result
        else:
            if condition in result.reference["Measurement Reference"].iloc[0]:
                filtered_dict[key] = result
    return filtered_dict

def split_df_by_key(df : pd.DataFrame, key_list : list):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        keys (list): _description_
    """
    key_dict = {key : df[df["reference"].str.contains(key)] for key in key_list}
    return key_dict

def get_energy_counts(rate_df, energy="thermal"):
    # return df of energy dependent counts depending on selection of either
    # thermal, epithermal or fast neutrons
    if energy == "thermal":
        df = rate_df[["rt(s)", "Det1", "Det2"]]
        df.loc[~(df==0).any(axis=1)]
        return df
    elif energy == "epithermal":
        df = rate_df[["rt(s)","Det3", "Det4"]]
        df.loc[~(df==0).any(axis=1)]
        return df
    elif energy == "fast":
        df = rate_df[["rt(s)","Det5", "Det6"]]
        df.loc[~(df==0).any(axis=1)]
        return df

def select_data_coordinate_range(data : dict[diamon], xmin : float, xmax : float, ymin : float, ymax : float):
    """selects data that fit into rectangular coordinate box

    Args:
        data (dict[diamon]): dictionary of diamon measurements
        xmin (float): minimum x coordinate
        xmax (float): maximum x coordinate
        ymin (float): minimum y coordinate
        ymax (float): maximum y coordinate

    Returns:
        dict: diamon dict inside valid coord
    """
    filtered_dict = {}
    for result in data.values():
        if (result.x > xmin) & (result.x < xmax) & (result.y > ymin) & (result.y < ymax):
            filtered_dict[result.file_name] = result
    return filtered_dict

def normalise_dose(data):
    """
    normalise dose measurement to the current of beam
    Args:
        data (obj): diamon class

    Returns:
        data: diamon class object with normalised dose for each time
    """
    if data.beamlines.target_station == "1":
        try:
            data.out_data["norm_dose"] = (140 * data.out_data["H*(10)r"].divide(data.out_data[data.beamlines.current_info]))
        except ZeroDivisionError:
            data.out_data["norm_dose"] = data.out_data["H*(10)r"]
        condition = (~np.isfinite(data.out_data["norm_dose"])) |(data.out_data["ts1_current"] < 120)
        data.out_data["norm_dose"] = np.where((condition) , data.out_data["H*(10)r"], data.out_data["norm_dose"])
    elif data.beamlines.target_station == "2":
        try:
            data.out_data["norm_dose"] = (35 * data.out_data["H*(10)r"].divide(data.out_data[data.beamlines.current_info]))
        except ZeroDivisionError:
            data.out_data["norm_dose"] = data.out_data["H*(10)r"]
        condition = (~np.isfinite(data.out_data["norm_dose"])) |(data.out_data["ts2_current"] < 25)
        data.out_data["norm_dose"] = np.where((condition) , data.out_data["H*(10)r"], data.out_data["norm_dose"])

    data.norm_dose = data.out_data["norm_dose"].tail(1)
    return data