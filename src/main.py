import diamon_read_data as dia
import diamon_analysis as da

location_path = "data\Measurement_location.csv"
fname = "shutters_pickle.pkl"

def main(selected_building):
    "selected building : 'any', 'ts1', 'ts2'"
    shutters = da.load_pickle(fname)
    diamon_path = da.select_location(selected_building)
    print(diamon_path)
    data = dia.read_diamon_folders(diamon_path, location_path)
    beam_data = {name: da.filter_shutters(dic, shutters) for name, dic in data.items()}
    beam_df = da.convert_to_df(beam_data)
    return beam_data, beam_df

if __name__ == "__main__":
    main("all")
