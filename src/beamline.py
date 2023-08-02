import pandas as pd

class beamline():
    """
    This class defines a neutron instrument information - the location of the
    measurement & what  beamlines it neighbours
    """
    def __init__(self, ref, df):
        self.target_station = ref[0]
        # for extracting correct channel name and normalisation
        if self.target_station == "1":
            self.current_info = "ts1_current"
        elif self.target_station == "2":
            self.current_info = "ts2_current"
        self.get_info(ref, df)

    def __str__(self):
        return self.name

    def get_info(self, ref : str, df : pd.DataFrame):
        """
        gets building and instrument beamline information for the measurement
        Args:
            ref (str): key that idenitifies building and beamline
            df (pd.DataFrame): df of key information
        """
        beam_info = df[df["key"] == ref[1:3]]
        #name of instrument
        self.name = beam_info.Name.values[0]
        #name of building
        self.set_building_position(beam_info)
        beam_df = df.xs(beam_info.index.get_level_values("Location")[0], level="Location").set_index("Name")
        self.all_neighbours = beam_df
        # try except if measurement not on a beamline eg: EPB
        try:
            self.closest_neighbours = self.set_neighbours(beam_df)
            
        except ValueError:
            print("error on beamline - no near neighbours")
        self.influx_data = self.all_neighbours.index.tolist()
        self.influx_data.append(self.current_info)

    def get_beam_info(self):
        # if measurement on beamline so only need current info for channel data
        self.name = "no_beamline"
        self.influx_data = [self.current_info]

    def set_neighbours(self, df : pd.DataFrame):
        """
        sets the neighbours next to the measurement from the target station csv
        Args:
            df (pd.DataFrame): df of all beamline info and keys

        Returns:
            df: rows of df matching neighbour key
        """
        idx = df.loc[self.name].Number
        return df[(df.Number == idx -1) | (df.Number == idx +1)]

    def set_building_position(self, beamline : pd.DataFrame):
        """set which target station measurement in (TS1/TS2)

        Args:
            beamline (pd.DataFrame): multi index df
        """
        self.building_position = beamline.index.get_level_values("Location")[0]

    @staticmethod
    def get_location(instrument , df : pd.DataFrame):
        """try excep to get the beam location

        Args:
            beam (beamline): 
            df (pd.DataFrame): 

        Returns:
            location (str): returns location
        """
        try:
            location = instrument.location
        except AttributeError:
            beamline.set_location_result(instrument, df)
            location = instrument.location
        except KeyError:
            print("no beamline")
        return location
