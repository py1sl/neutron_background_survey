#from datetime import timedelta, datetime
import datetime
import json
import warnings
import re
import matplotlib.pyplot as plt
import pytz
from influx_data_utils.process_influx_data.datetime_localiser import \
    DatetimeLocaliser
from influx_data_utils.process_influx_data.influx_querier import InfluxQuerier
from influxdb_client import InfluxDBClient
# remove warnings about a pivot function
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter('ignore', MissingPivotFunction)

class query_object:

    def __init__(self, start, end, names=None):
        if channel_names is None:
            self.names = channel_names()
        else:
            self.names = names
        self.utc = pytz.timezone("UTC")
        self.bst = pytz.timezone("Europe/London")
        self.start = self.date_to_str(start)
        self.end = self.date_to_str(end)
        self.data = {}
        self.cycles = {"22_03": [datetime.date(2022,9,13), 
                                 datetime.date(2020,10,14)], 
                       "22_04": [datetime.date(2022,11,8), 
                                 datetime.date(2022,12,16)],
                       "22_05": [datetime.date(2022,2,7),
                                 datetime.date(2022,3,31)]
                       }

    def __str__(self):
        return self.names

    def date_to_str(self, datetime_obj):
        localiser = DatetimeLocaliser(self.bst, self.utc)
        date = localiser.convert_datetime_to_string(datetime_obj)
        return date

    def str_to_date(self, str_date: str):
        localiser = DatetimeLocaliser(self.bst, self.utc)
        date = localiser.convert_string_to_datetime(str_date)
        return date

    @staticmethod
    def get_data(start, end, names):
        query = query_object(start, end, names)
        query.data = query.influx_query(names, query.start, query.end)
        return query

    def last_query(self, channel, start_str, end_str):
        client_query_api, influx_querier = self.load_client_api()
        df = InfluxQuerier.query_influx(client_query_api, channel_name=channel, 
                                        start_time=start_str, end_time=end_str).tail(1)
        if df.empty:
            #case returns empty - no update since start
            start_time = self.str_to_date(start_str)
            prev_week = (start_time - datetime.timedelta(days=1))
            new_start_time = self.date_to_str(prev_week)
            df = self.last_query(channel, new_start_time, start_str)
        else:
            df = df.drop(['result','table'], axis=1)
            return df

    def influx_query(self, channels, start_str, end_str):
        client_query_api, influx_querier = self.load_client_api()
        shutters_data = {}
        for channel in channels:
            df = InfluxQuerier.query_influx(
                client_query_api, channel_name=channel, 
                start_time=start_str,
                end_time=end_str
            )
            if df.empty:
                #case returns empty - no update since start
                start_time = self.str_to_date(start_str)
                prev_week = (start_time - datetime.timedelta(days=1))
                new_start_time = self.date_to_str(prev_week)
                df = self.last_query(channel, new_start_time, start_str)
            else:
                df = df.drop(['result','table'], axis=1)
            name = re.search("(?<=::)(.*?)(?=:)", channel).group()
            if name == 'e1':
                name = 'zoom'
            elif name == 'e6':
                name = 'larmor'
            elif name == 'w5':
                name = 'imat'
            if channel == "local::beam:target2":
                name = "ts2_current"
            elif channel == "local::beam:target":
                name = "ts1_current"
            shutters_data[name] = df
        return shutters_data

    def load_client_api(self):
        with open("influx.auth", 'r') as f: # create a json file with an "AUTH_TOKEN" key
            auth_file = json.load(f)
        auth_token = auth_file["AUTH_TOKEN"]
        client = InfluxDBClient(url="https://infra.isis.rl.ac.uk:8086", token=auth_token, org="4298498d740c3795")
        client_query_api = client.query_api()
        influx_querier = InfluxQuerier(client_query_api, source_timezone=self.utc, target_timezone=self.bst)
        return client_query_api, influx_querier

    def get_shutter_status(self, time):
        return

def channel_names():

    channels = ["t2shut::chipir:status", "t2shut::sans2d:status", "t2shut::wish:status", 
                "t2shut::inter:status", "t2shut::offspec:status", "t2shut::let:status", 
                "t2shut::nimrod:status", "t2shut::polref:status", "t2shut::e1:status", "t2shut::e6:status", 
                "t2shut::w5:status", "local::beam:target", "local::beam:target2"]
    return channels
"""
start = [2022,11, 8,0, 0,0]
end = [2022, 12, 16, 23, 59, 59]
names = channel_names()
data = query_object.get_data(names, start, end)
print(data)
"""
def date_to_str( datetime_obj):
    utc = pytz.timezone("UTC")
    bst = pytz.timezone("Europe/London")
    localiser = DatetimeLocaliser(bst, utc)
    date = localiser.convert_datetime_to_string(datetime_obj)
    return date