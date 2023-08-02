# functions that dont seem to have a use anymoe but may in the future



def get_initial_date(start, shutters):
    beam_df = pd.read_csv("data/target_station_data.csv", index_col=["Building", "Name"])
    end = get_date_df(shutters, "ts2_current", False)
    end = end.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    start = datetime(2022, 10, 1)
    start = idb.date_to_str(start)
    shutter = influx_db_query([start, end], beam_df.channel_name, update=False)
    comb_shutter = append_new_shutter_info(shutter, shutters)
    return comb_shutter
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

def select_shutter(self, selected_shutter):
    if self.name == "no_beamline":
        names = None
    elif selected_shutter == "all":
        names = self.all_neighbours.index
    elif selected_shutter == "closest":
        names = self.closest_neighbours.index
        names = np.append(names, self.name)
    elif selected_shutter == "own":
        #selected just the own
        names = self.name
    elif selected_shutter == "none":
        names = None
    return names
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