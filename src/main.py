import diamon_read_data as dia
import diamon_analysis as da

ts2_path = "data/ts2_measurements/DIAMON*"
location_path = "data/measurement_location.csv"
fname = "shutters_pickle.pkl"

def main():
    shutters = da.load_pickle(fname)
    data = dia.read_diamon_folders(ts2_path, location_path)
    beam_data = {name: da.filter_shutters(dic, shutters) for name, dic in data.items()}
    beam_df = da.convert_to_df(beam_data)
    return beam_data, beam_df

if __name__ == "__main__":
    main()
