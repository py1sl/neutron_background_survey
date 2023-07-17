#from datetime import timedelta, datetime
import datetime
import json
import warnings
import pytz
import pandas as pd
from influx_data_utils.process_influx_data.datetime_localiser import \
    DatetimeLocaliser
from influx_data_utils.process_influx_data.cycle_datetime_formatter import CycleDateTimeFormatter
from influx_data_utils.process_influx_data.influx_data_processor import InfluxDataProcessor, DataProcessingConfig
from influx_data_utils.process_influx_data.influx_querier import InfluxQuerier
from influxdb_client import InfluxDBClient
# remove warnings about a pivot function
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter('ignore', MissingPivotFunction)

class cycle_dict:
    """_summary_:
    This class defines all user cycles as datetimes under a dict object
    """
    def __init__(self):
        self.cycles = {"22_01": [datetime.date(2022,5, 3), datetime.date(2022,6, 2)],
                "22_02": [datetime.date(2022,6, 28), datetime.date(2022,7, 29)],
                "22_03": [datetime.date(2022,9,13), datetime.date(2022,10,14)], 
                "22_04": [datetime.date(2022,11,8), datetime.date(2022,12,16)],
                "22_05": [datetime.date(2023,2,7), datetime.date(2023,3,31)],
                "23_01": [datetime.date(2023,4, 25),  datetime.date(2023,5,26)],
                "23_02": [datetime.date(2023,6,27), datetime.date(2023,8,4)],
                "23_03": [datetime.date(2023,9,19), datetime.date(2023,10,20)],
                "23_04": [datetime.date(2023,11, 14), datetime.date(2023, 12, 19)],
                "23_05": [datetime.date(2023,2,13), datetime.date(2023,3,22)]}
    
    def get_cycle(self, keys : list[str]):
        """
        This function gets selected cycles from a list of cycle strings
        arg: keys: (list[str]) keys of beam cycles to get data
        returns: dict of self.cycles with matching keys
        """
        filtered_cycles = {key: value for key, value in self.cycles.items() if key in keys}
        return filtered_cycles

class query_object():
    """
    This class initiates a query object to pull from influx db.
        channels: list of selected channels
        start, end: str of start and end times
        condition: "condition to filter out selected data"
        remove_empty_entries: during period of missing data set to remove data
        timezone: selected timezone of data to access from
    """
    def __init__(self, channels: list, timezone: str ="Europe/London"):
        self.channels = channels
        self.utc = pytz.timezone("UTC")
        self.timezone = pytz.timezone(timezone)
        self.raw_dfs : list[pd.DataFrame] = []
        self.filtered_df : dict[dict[pd.DataFrame]] = None
        self.selected_cycles : dict[datetime.datetime] = None
        self.start : str = None
        self.end : str = None
        self.data_processing_config : DataProcessingConfig = None

    def __str__(self):
        return self.channels

    def cycle_formatter(self, cycles : dict[list[datetime.datetime]]):
        """converts cycle dictionary into datepair strings
        Args:
            cycles : dict{list[datetime.datetime]} - dictionary containing 
              list of start and end of cycles
        Returns:
            string_datetime_pairs : list of datepair strings corresponding to each datetime given, 
              localised to the correct timezone. 
        """
        _, datetimes = CycleDateTimeFormatter.convert_cycles_to_daily_dates(cycles)
        string_datetime_pairs = CycleDateTimeFormatter.convert_daily_datepairs_to_datepair_strings(
            datetimes, source_date=self.timezone)
        return string_datetime_pairs

    def merge_adjacent_days_of_data(self):
        """Concatenate dataframes that directly follow eachother, creating a single DF of continuous data. 

        Args:
            dfs (list): List of lists of dataframes for each day and each channel queried.
        Returns:
            list: list of lists of continuous dataframes.
        """
        cont_dfs = {cycle: {} for cycle in self.selected_cycles.keys()}
        for dfList in (self.dfs()):
            # save the date of the current dataframe
            df =  pd.concat(dfList)
            for key, dates in self.selected_cycles.items():
                start =  dates[0] - datetime.timedelta(days=1)
                end =  dates[1] + datetime.timedelta(days=1)
                cont_dfs[key][self.channels] = df[start:end]
        return cont_dfs

    def influx_query_cycle(self):
        """
        use api influx querir to grab df from influx db. cycledatetimeformatter with influx querier and 
        fully process raw data functions to get df for each adjacent days in cycles
        
        """
        string_datetime_pairs = self.cycle_formatter(self.selected_cycles)
        influx_querier = self._load_client_api()
        self.raw_dfs = influx_querier.query_all_data_over_ranges(self.channels, string_datetime_pairs)
        self.filtered_df = self.merge_adjacent_days_of_data()

        # TO DO: fix ACCEL CONTROL PROCESS DATA (CONDITION & EMPTY)
        # influx_data_processor = InfluxDataProcessor(self.data_processing_config, minutes_of_missing_data_to_filter=60)
        # self.final_dfs = influx_data_processor.fully_process_raw_data(self.dfs)

    def influx_query_datetime(self):
        """This query method uses two sets of datetimes

        Returns:
            df dict: dict of df for each channel loaded between two dt
        """
        # try localise start and end
        influx_querier = self._load_client_api()
        self.filtered_df = {channel: self._query_db(influx_querier, channel, self.start, self.end) for channel in self.channels}

    def _load_client_api(self):
        """
        opens api using auth token and initialises InfluxQuerirer class
        Returns:
            influx_querier (InfluxQuerier): object initialiased to pull from db
        """
        with open("src\\influx.auth", 'r') as f: # create a json file with an "AUTH_TOKEN" key
            auth_file = json.load(f)
        auth_token = auth_file["AUTH_TOKEN"]
        client = InfluxDBClient(url="https://infra.isis.rl.ac.uk:8086", token=auth_token, org="4298498d740c3795")
        client_query_api = client.query_api()
        influx_querier = InfluxQuerier(client_query_api, source_timezone=self.utc, target_timezone=self.timezone)
        return influx_querier

    def _query_db(self, influx_querier : InfluxQuerier, channel : str, start : str, end : str):
        """query db function between two times

        Args:
            influx_querier (InfluxQuerier): influxquerier object
            channel (str): string of channel name to query db
            start (str): start time
            end (str): end time

        Returns:
            df: pd.DataFrame
        """
        df = influx_querier.query_influx(influx_querier.client_query_api, channel_name=channel, 
                                        start_time=start, end_time=end)
        if df.empty:
            end = str_to_date(start)
            prev_day = (end - datetime.timedelta(days=1))
            start = date_to_str(prev_day)
            df = self._query_db(influx_querier, channel, start, self.start).tail(1)
        else:
            df = df.drop(['result','table'], axis=1)
        return df

    @staticmethod
    def check_cycle(date, cycles):
        """

        Args:
            date (datetime): datetime object at date of recording
            cycles (list[datetime]): list of 2 datetimes

        Returns:
            list[datetime] : list of cycle start and end
        """
        for cycle in cycles.values():
            if cycle[0] < date < cycle[1]:
                return cycle

    @staticmethod
    def get_data_datetime(start : datetime.datetime, end : datetime.datetime, channels : list[str], 
                          timezone : str = "Europe/London"):
        """
        Static class method 
        Args:
            start (datetime): start of date range
            end (datetime): end of date range

        Returns:
            query object: query object containing df and key info
        """
        query = query_object(channels=channels, timezone=timezone)
        query.start = date_to_str(start, timezone)
        query.end = date_to_str(end, timezone)
        query.influx_query_datetime()
        return query

    @staticmethod
    def get_data_cycle(cycles : list[str], channels : list[str], timezone : str="Europe/London",
                       condition=lambda dataframe : False, remove_empty_entries: bool =False):
        """

        Args:
            cycles (list[str]): list of cycles to call data between
            channels (list[str]): list of channels to get from influx db
            timezone (str, optional): timezone to use - default is europe/london
            condition (df bool, optional): once df is extracted filter based on condition;
            -----example of this is- 'lambda dataframe : dataframe["_value"] > 10'. Defaults to lambda dataframe:False.
            remove_empty_entries (bool, optional): whether to remove blank entries - no data found on day. Defaults to False.

        Returns:
            query_object: object query containing key info and df
        """
        query = query_object(channels=channels, timezone=timezone)
        query.selected_cycles = cycle_dict().get_cycle(cycles)
        query.data_processing_config = [DataProcessingConfig(
            channel, filter_condition=condition, remove_empty_periods=remove_empty_entries) for channel in channels]
        query.influx_query_cycle()
        return query

    def merge_adjacent_days_of_data(self):
        """Concatenate dataframes that directly follow eachother, creating a single DF of continuous data. 

        Args:
            dfs (list): List of lists of dataframes for each day and each channel queried.
        Returns:
            cycle_dic (dict): nested dict containing df for each selected shutter for each selected cycle
        """
        cycle_dic = {key: {channel : None for channel in self.channels} for key in self.selected_cycles.keys()}
        channel_dic = {channel : pd.concat([df[i] for df in self.raw_dfs]) for i, channel in enumerate(self.channels)}
        for channel, df in channel_dic.items():
            # save the date of the current dataframe
            for key, dates in self.selected_cycles.items():
                start =  dates[0] - datetime.timedelta(days=1)
                end =  dates[1] + datetime.timedelta(days=1)
                cycle_dic[key][channel] = df[start:end]
        return cycle_dic

def date_to_str(datetime_obj : datetime.datetime, source_timezone : str, target_timezone : str ="UTC"):
    """
    localise datetime to target timezone
    Args:
        datetime_obj (datetime.datetime): datetime

    Returns:
        date: 
    """
    source_timezone = pytz.timezone(source_timezone)
    target_timezone = pytz.timezone(target_timezone)
    localiser = DatetimeLocaliser(source_timezone, target_timezone)
    date = localiser.convert_datetime_to_string(datetime_obj)
    return date

def str_to_date(str_date: str, source_timezone, target_timezone="UTC"):
    """converts a date to the query set timezone

    Args:
        str_date (str): 

    Returns:
        date (datetime): datetime object in self.timezone
    """
    localiser = DatetimeLocaliser(source_timezone, target_timezone)
    date = localiser.convert_string_to_datetime(str_date)
    return date