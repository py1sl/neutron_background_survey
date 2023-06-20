#from datetime import timedelta, datetime
import datetime
import json
import warnings
import numpy as np
from collections import namedtuple as nt
import pytz
from influx_data_utils.process_influx_data.datetime_localiser import \
    DatetimeLocaliser
from influx_data_utils.process_influx_data.influx_querier import InfluxQuerier
from influxdb_client import InfluxDBClient
# remove warnings about a pivot function
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter('ignore', MissingPivotFunction)

class query_object:

    def __init__(self, names=None, current_data=False):
        if names is not None:
            self.names = names.values
            if current_data is True:
                self.names = np.append(self.names, ["local::beam:target", "local::beam:target2"])
        self.utc = pytz.timezone("UTC")
        self.bst = pytz.timezone("Europe/London")
        self.start = None
        self.end = None
        self.data = {}
        self.cycles = {"22_03": [datetime.date(2022,9,13), 
                                 datetime.date(2020,10,14)], 
                       "22_04": [datetime.date(2022,11,8), 
                                 datetime.date(2022,12,16)],
                       "22_05": [datetime.date(2023,2,7),
                                 datetime.date(2023,3,31)],
                       "23_01": [datetime.date(2023,4, 25), 
                                 datetime.date(2023,5,26)],
                       "23_02": [datetime.date(2023,6,27),
                                 datetime.date(2023,8,4)]
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
    def get_data(start, end):
        query = query_object()
        query.start = query.date_to_str(start)
        query.end = query.date_to_str(end)
        query.data = query.influx_query()
        return query

    def last_query(self, channel, start_str, end_str):
        client_query_api, influx_querier = self.load_client_api()
        df = InfluxQuerier.query_influx(client_query_api, channel_name=channel, 
                                        start_time=start_str, end_time=end_str).tail(1)
        if df.empty:
            #case returns empty - no update since start
            start_time = self.str_to_date(start_str)
            prev_week = (start_time - datetime.timedelta(weeks=1))
            new_start_time = self.date_to_str(prev_week)
            return self.last_query(channel, new_start_time, start_str)
        else:
            df = df.drop(['result','table'], axis=1).set_index("_time")
            return df

    def influx_query(self, channel_names, update=False, timezone="Europe/London"):
        channels = self.names
        start_str = self.start
        end_str = self.end
        client_query_api, influx_querier = self.load_client_api()
        idb_data = {}
        for channel in channels:
            print(channel  + " read in  at " + str(start_str) + "\n")
            df = InfluxQuerier.query_influx(
                client_query_api, channel_name=channel, 
                start_time=start_str,
                end_time=end_str
                )
            if df.empty == True:
                #case returns empty - no update since start
                if update == False:
                    start_time = self.str_to_date(start_str)
                    prev_day = (start_time - datetime.timedelta(days=1))
                    new_start_time = self.date_to_str(prev_day)
                    df = self.last_query(channel, new_start_time, start_str)
            elif df.empty == False:
                df = df.drop(['result','table'], axis=1).set_index("_time")
                df.index = df.index.tz_convert(timezone)
            if channel == "local::beam:target":
                name = "ts1_current"
            elif channel == "local::beam:target2":
                name = "ts2_current"
            else:
                name = channel
            idb_data[name] = df.sort_index()
        return idb_data

    def query_db(self):
        channel = self.name
        start = self.start
        end = self.start
        client_query_api, influx_querier = self.load_client_api()
        df = InfluxQuerier.query_influx(client_query_api, channel_name=channel, 
                                        start_time=start, end_time=end)
        if df.empty:
            end = self.str_to_date(start)
            prev_week = (end - datetime.timedelta(days=1))
            start = self.date_to_str(prev_week)
            df = self.last_query(channel, start, end)
        else:
            df = df.drop(['result','table'], axis=1)
        return df

    def load_client_api(self):
        with open("src\\influx.auth", 'r') as f: # create a json file with an "AUTH_TOKEN" key
            auth_file = json.load(f)
        auth_token = auth_file["AUTH_TOKEN"]
        client = InfluxDBClient(url="https://infra.isis.rl.ac.uk:8086", token=auth_token, org="4298498d740c3795")
        client_query_api = client.query_api()
        influx_querier = InfluxQuerier(client_query_api, source_timezone=self.utc, target_timezone=self.bst)
        return client_query_api, influx_querier

    def get_shutter_status(self, time):
        return

def date_to_str( datetime_obj):
    utc = pytz.timezone("UTC")
    bst = pytz.timezone("Europe/London")
    localiser = DatetimeLocaliser(bst, utc)
    date = localiser.convert_datetime_to_string(datetime_obj)
    return date