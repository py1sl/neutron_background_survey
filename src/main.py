import src.diamon_read_data as drd
import src.shutter_analysis as sa
import src.diamon_analysis as da
import pandas as pd
import numpy as np
#------MAIN PROGRAM-------#
# This file will load all relevent data and information to be used for analysis

def main(save=False):
    """
    Start of program. Controls the reading in of diamon detector measurements and shutter data
    from the InfluxDB. Option to save loaded data for future use
    Args:
        save (bool) . option to save diamon data. default is False.
    Returns:
        dict[diamon]: dictionary of initialised diamon
    """
    main_data = drd.read_data()
    start = drd.get_earliest_entry(main_data)
    end = drd.get_last_entry(main_data)
    beamline_df = pd.read_csv("data/target_station_data.csv", index_col=["Building", "Name"])
    channel_names = beamline_df.channel_name.dropna().to_numpy()
    channel_names = np.append(channel_names, ["local::beam:target", "local::beam:target2"])
    channel_data = sa.load_channel_data(start, end, channel_names).filtered_df
    main_data = {key: sa.filter_shutters(result, channel_data) for key, result in main_data.items()}
    # save
    if save == True:
        da.save_pickle(main_data, "diamon_data")
        da.save_pickle(main_data, "shutter_data")
    return main_data

def shutter_df(data, selected_shutter, bb):
    "return panda df showing selected shutter status for data"
    df = da.filter_shutter_status(data, selected_shutter, bb)
    open_shutters = da.flag_shutter(df, selected_shutter, True)
    closed_shutters = da.flag_shutter(df, selected_shutter, False)
    open_shutters = da.find_abs_error(open_shutters, "norm_dose", "Hr_un%")
    closed_shutters = da.find_abs_error(closed_shutters, "norm_dose", "Hr_un%")
    return {"data":data, "df":df, "open-shutter":open_shutters, "closed-shutter":closed_shutters}

if __name__ == "__main__":
    main()
