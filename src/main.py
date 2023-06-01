import src.diamon as d
import src.diamon_analysis as da
import src.shutter_analysis as sa

def main(save=False):
    """_summary_

    Returns:
        _type_: _description_
    """
    shutter_data = sa.load_shutter_data()
    main_data = d.read_data(shutter_data)
    main_data = {key: sa.filter_shutters(result, shutter_data) for key, result in main_data.items()}
    # save
    if save == True:
        da.save_pickle(main_data, "diamon_data")
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
