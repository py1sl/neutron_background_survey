import numpy as np
import pandas as pd
#issue with scipy - cant import lib?
from scipy.signal import find_peaks
import pickle
import src.influx_data_query as idb
from datetime import datetime
import dask.dataframe as dd

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
    match building:
        case "TS1":
            ts1_dict = {key: dic for key, dic in data.items() if (
                dic["reference"]["Measurement Reference"].iloc[0][0] == '1')} 
            return ts1_dict
        case "TS2":
            ts2_dict = {key: dic for key, dic in data.items() if (
                dic["reference"]["Measurement Reference"].iloc[0][0] == '2')}
            return ts2_dict

def ts1_beamline_names():
    """return list of beamlines in ts1
    """
    return []
def ts2_beamline_names():
    """
    returns a list of beamline names in ts2
    """
    return ["chipir", "sans2d", "wish", 
                "inter", "offspec", "let", 
                "nimrod", "polref", "zoom", "larmor", 
                "imat", "epb"]
def names():
    """
    returns a list of beamline names in ts2
    """
    return ["C", "S", "W", 
                "T", "O", "E",
                "N", "P", "Z", "L", 
                "I", "B"]
    
#ToDO: needs tidying up
def convert_to_df(data):
    """
    returns a dictionary of dataframes with keys for each
    instrument beamline. if on epb = epb
    
    """
    beam_df = {}
    beamline_names = names()
    for i, name in enumerate(beamline_names):
        series_list = []
        for result in data.values():
            ref = result.reference["Measurement Reference"].iloc[0]
            if ref[1] == beamline_names[i]:
                series_list.append(convert_to_ds(result))
        beam_df[name] = pd.DataFrame(series_list)
        if not beam_df[name].empty:
            beam_df[name] = beam_df[name].sort_values("distance")
    return beam_df

def convert_to_ds(data):
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
    s1 = get_distance(s1)
    return s1

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
    s1 = get_distance(s1)
    return s1

def get_distance(data, dimension=2):
    """
    get 2d and 3d pythag distance between coordinates and the origin
    Args:
        data (panda series): series containing diamon data
        dimension (int, optional): 2d or 3d dimension. Defaults to 2.

    Returns:
        data (series): data with distance column added
    """
    if dimension == 2:
        data['distance'] = np.sqrt(data['x']**2 + data['y']**2)
    elif dimension == 3:
        data['distance'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    return data

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
    energy = data.energy_bin
    flux = data.flux_bin
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
    return str(np.datetime_as_string(dt))#

def get_east_west_names(reference):
    ref = reference["Measurement Reference"].iloc[0]
    if ref[0] == "1":
        return
    
    elif ref[0] == "2":
        if ref[1] == "C" or ref[1] == "N" or ref[1] == "I" or ref[1] == "E":
            return ["chipir", "nimrod", "imat", "let"]
        elif ref[1] == "W" or ref[1] =="L" or ref[1] ==ref[1] =="O" or ref[1] =="P" or ref[1] =="S" or ref[1] =="Z" or ref[1] == "T":
            return ["wish", "larmor", "offspec", "inter", "polref", "sans2d", "zoom"]
        elif ref[1] == "B":
            return ["beamline"]
        else:
            "ts1 or beamline"
            return []

def get_beamline_name(ref):
    """
    get building and beamline name as list of str
    Args:
        data (df): df of measurement location reference

    Returns:
        list: str
    """
    ts1 = {"sandals": "Sa", "prisma": "Pr", "surf": "Su",
           "crisp": "Cr", "loq": "Lq", "iris": "Is", "polaris":"Pl",
           "tosca": "Ts", "Het": "Ht", "pearl": "Pe", "hrpd": "Hr", "engin-x": "Ex",
           "gem": "Gm", "Mari": "Me", "sxd":"Sx", "vesuvio": "Vs", "maps": "Ms",
           "chronus": "Ch", "argus": "Ar", "musr": "Mu", "hifi": "Hi", "epb":"BL"}
    ts2 = {"epb":"B", "chipir": "C", "imat": "I", "inter": "T", "larmor": "L", "let":"E", 
           "nimrod":"N", "offspec":"O", "polref":"P", "sans2d":"S", "wish":"W", "zoom":"Z"}
    #extracts a code reference format : 2ZT-3 : ts2, zoom, top, 3rd measurement

    if ref[0] == "1":
        match = next((key for key, value in ts1.items() if ref[1:3] in value), "null")
        building = "ts1"
    elif ref[0] == "2":
        match = next((key for key, value in ts2.items() if ref[1] in value))
        building = "ts2"
    else:
        print("No building match found! ")
        return
    return [building, match]

def filter_shutters(data, shutters):
    """
    filter all shutter query info for beamline shutters and current

    Args:
        data (object): diamon data
        shutters (dict): shutter df

    Returns:
        dict: diamon data with out data having a current and shutter status at each timestep
    """
    #get beamline and building names for data being measured
    shutter_list = get_east_west_names(data.reference)
    # if on the epb no shutter info - not near a beamline only get current
    if len(shutter_list) > 1:
        for name in shutter_list:
            sel_shutter = shutters[name].sort_index()
            data.out_data[name] = [get_query_info(sel_shutter, time) for time in data.out_data["datetime"]]
        data.out_data["shutter-open"] = data.out_data[data.beamline]
    #call function to extract ts1 and ts2 current
    data.out_data["ts2_current"] = get_current_info(data, shutters["ts2_current"])
    data.out_data["ts2_current"] = data.out_data["ts2_current"].fillna(0)
    #normalise dose to the current
    data = normalise_dose(data)
    return data

def get_query_info(data, time):
    """
    get shutter status at most recent time since the selected time
    Args:
        data (df): shutter df indexed by datetime
        time (datetime object): time of recordingdatetime

    Returns:
        boolean: true if shutter open (2), false if shutter closed/setup (1/3)
    """
    #extract the tail of df where shutter df matches previous times
    #print(data.tail(4))
    status = data.loc[:time].tail(1)["_value"].values[0]
   #status = data[data["_time"] < time].tail(1)["_value"].values[0]
    if status == 1:
        return True
    if status == 2:
        return False
    if status == 3:
        return False
    
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
# TODO: fix dose when 0 current 
def normalise_dose(data):
    """
    normalise dose measurement to the current
    Args:
        data (dict): dict of measurement data

    Returns:
        dict: data with extra col for normalised dose
    """
    target_station = get_beamline_name(data.reference["Measurement Reference"].iloc[0])[0]
    if target_station == "ts1":
        #divide by mean current at the time and for 0 current set dose to 0
        # 30 is average beam current for day
        data.out_data["norm_dose"] = (160 * data.out_data["H*(10)r"].divide(data.out_data["ts1_current"]).replace(np.inf, 0))
    if target_station == "ts2":
        data.out_data["norm_dose"] = (35 * data.out_data["H*(10)r"].divide(data.out_data["ts2_current"]).replace(np.inf, 0))
    return data

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

#old function
def find_spect_peaks(data):
    """
    find the peaks for every data
    Args:
        data (list): _description_

    Returns:
        2 lists:
    """
    energy_list = []
    flux_list = []
    for spectra in data:
        energy, flux = peaks_finder(spectra)
        energy_list.append(energy)
        flux_list.append(flux)
    return energy_list, flux_list

def check_east_west(name):
    west = ["chipir", "nimrod", "imat", "let"]
    east = ["wish", "larmor", "offspec", "inter", "polref", "sans2d", "zoom"]
    if name in west:
        return "west"
    elif name in east:
        return "east"
    else:
        #beamline
        return "beam"

def filter_shutter_status(data, selected_shutter, bb=False):
    """
    for each result get a df
    filter the dataframe by open or closed shutter
    """
    filtered_df = []
    for result in data.values():
        #remove low current
        result = filter_low_beam_current(result, 25)
        #remove epb measurements with no shutter status
        # check has a valid shutter
        building, name = get_beamline_name(result.reference['Measurement Reference'].iloc[0])
        result_location = get_east_west_names(result.reference)
        if name == "epb":
            #take the last line of out data
            out = result.out_data.iloc[-1]
            dseries = (convert_status_to_ds(result))
            filtered_df.append(pd.concat([dseries, out]))
        elif building == "ts1":
            continue
        elif selected_shutter == "own":
            print(result.file_name)
            name = "shutter-open"
            filtered_df = filtered_df + (last_row_shutter_change(result, name))
        elif name in selected_shutter:
            for name in selected_shutter:
                filtered_df = filtered_df + (last_row_shutter_change(result, name))

    df = pd.DataFrame(filtered_df)
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
    data = data[(result.out_data["t(s)"] > 1000) | (result.out_data["Hr_un%"] < 20)]
    #get df of data when shutter is open
    df = data[shutter_name]
    #get boolean for index of row where change in status and always get last value in df
    filter = (df.ne(df.shift(-1)))
    last = df.tail(1)
    change_times = data[(filter) | (last)]
    filtered_list  = []
    for _, row in change_times.iterrows():
        dseries = convert_status_to_ds(result)
        filtered_list.append(pd.concat([dseries, row]))
    return filtered_list

def flag_shutter(data, shutter, flag=True):
    """
    boolean mask of data to get data matching selected flag. flag is true/false for open/closed
    """
    if shutter == "own":
        shutter = "shutter-open"
    data = data[(data[shutter] == flag) | (data[shutter].isna())]
    averaged_data = average_repeated_data(data, shutter).dropna(subset = ['x', 'y', 'norm_dose']).reset_index()
    return averaged_data

#error when doing this
def average_repeated_data(df, shutter):
    """
    When measurement has multiple data for same date and location take average of data
    Args:
        df (dataframe): key information for data in df
    returns: filtered df with averages taken for repeats
    """
    keep = df[["key", "reference", "start", "end", shutter]].drop_duplicates()
    filtered_df = df.groupby("key").mean(numeric_only=True)
    averaged_df = pd.merge(filtered_df, keep, on="key")
    return averaged_df

def filter_low_beam_current(data, minimum_current):
    """
    This function removes any data in out iles where beam current less than the argument minimum current
    Args:
        data (dict): dict of all data information
        minimum_current (float): minimum current to include

    Returns:
        dict: same data with data at a time with current < minimum removed
    """
    data.out_data= data.out_data[data.out_data["ts2_current"] > minimum_current]
    return data

#Todo - add comparison between repeats of data  and return the key(same x,y,z)
def find_repeats(data):
    #check likewise data
    ref = list(set(data["reference"].values))
    filtered_df_dic = {}
    for text in ref:
        groups = text.split('-')
        joined_str = '-'.join(groups[:2]), '-'.join(groups[2:])
        if joined_str[1] != '':
            filtered_df_dic[joined_str[0]] = (data[data["reference"].str.match(joined_str[0])]).drop_duplicates()
    return filtered_df_dic

def group_repeated_results(data):
    keys = find_repeats(data)
    return keys

def get_x_y_z(data1, data2):
    open_plot_df = data1[['x', 'y',"norm_dose", "abs_error", 'Ther%', "Epit%", "Fast%"]]
    closed_plot_df = data2[['x', 'y',"norm_dose", "abs_error", 'Ther%', "Epit%", "Fast%"]]
    shutter_dict = {"open": open_plot_df, "closed": closed_plot_df}
    x = [shutter_dict["open"]['x'], shutter_dict["closed"]['x']]
    y = [shutter_dict["open"]['y'], shutter_dict["closed"]['y']]
    #look at dose vs abs error
    z = [shutter_dict["open"]['norm_dose'], shutter_dict["closed"]['norm_dose']]
    return x, y, z
def get_out_from_name(data):
    return

def check_updated_shutter_info(shutters, beam_df):
    # add check to load new data
    date = get_date_df(shutters, "ts2_current", "_time")
    if date.date() < datetime.today().date():
        print("Updating shutter information \n")
        shutters = append_new_shutter_info(shutters, beam_df)
        #saves new shutter information into pickle for later use
        save_pickle(shutters, "shutter_data")
        print("saved new data and loaded into program \n")
    return shutters

def get_date_df(df, channel_name, colname):
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
    new = latest_shutters(shutters, beam_df)
    result = {}
    for key, df in new.items():
        if key in shutters.keys():
            result[key] = pd.concat([shutters[key], df])
        else:
            result[key] = df
    return result

def latest_shutters(current_shutter, beam_df):
    last_time = get_date_df(current_shutter,"ts2_current", "_time")
    last_time = last_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    today = idb.date_to_str(datetime.today())
    dates = [last_time, today]
    shutters = influx_db_query(dates, beam_df.channel_name)
    return shutters

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
    filtered_df = df[ (df[shutter_names].all(axis=1)) & ~(df[shutter_names[0]].isna())]
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

def find_neighbours(beamline):
    
    # find what beamline performed on
    beamline_neighbours = {"nimrod":["let"], "let": ["nimrod", "imat"], "imat": ["let", "chipir"], "chipir":["imat"], 
                   "wish":["larmor"], "larmor":["wish", "offspec"], "offspec":["larmor", "inter"], 
                    "inter":["offspec", "polref"], "polref":["inter", "sans2d"], 
                    "sans2d":["polref", "zoom"], "zoom":["sans2d"]}
    for neighbour in beamline_neighbours.keys():
        if neighbour == beamline:
            neighbours = beamline_neighbours[neighbour]
            return neighbours

