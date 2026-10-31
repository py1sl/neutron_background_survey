from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import src.diamon as diamon
import src.diamon_read_data as diamon_read_data
import src.main as main


class IncrementalLoadingTests(unittest.TestCase):
    def test_update_diamon_data_merges_new_measurements(self):
        existing_data = {
            "old": SimpleNamespace(start_time=datetime(2023, 1, 1, tzinfo=timezone.utc))
        }
        new_data = {"new": SimpleNamespace()}
        folders = [Path("measurement")]

        with (
            patch.object(diamon_read_data.glob, "glob", return_value=["measurement/counters.txt"]),
            patch.object(diamon_read_data, "read_unloaded_file", return_value=folders[0]),
            patch.object(diamon_read_data, "load_diamon", return_value=new_data) as load_diamon,
        ):
            result = diamon_read_data.update_diamon_data(existing_data, pd.DataFrame())

        self.assertIs(result, existing_data)
        self.assertEqual(set(result), {"old", "new"})
        load_diamon.assert_called_once_with(unittest.mock.ANY, folders)

    def test_load_diamon_accepts_explicit_folder_list(self):
        folders = ["measurement-one", "measurement-two"]
        measurements = [
            SimpleNamespace(file_name="one"),
            SimpleNamespace(file_name="two"),
        ]

        with (
            patch.object(diamon_read_data.glob, "glob") as glob,
            patch.object(diamon_read_data.diamon, "diamon", side_effect=measurements),
            patch.object(diamon_read_data, "match_id_location", side_effect=lambda _, data: data),
        ):
            result = diamon_read_data.load_diamon(pd.DataFrame(), folders)

        glob.assert_not_called()
        self.assertEqual(result, {"one": measurements[0], "two": measurements[1]})


class MainCacheTests(unittest.TestCase):
    def test_main_saves_shutter_channels_to_their_own_cache(self):
        measurements = {"measurement": object()}
        channel_data = {"channel": pd.DataFrame()}
        saved = {}

        with (
            patch.object(main.drd, "read_data", return_value=measurements),
            patch.object(main.drd, "get_earliest_entry", return_value=None),
            patch.object(main.drd, "get_last_entry", return_value=None),
            patch.object(main.pd, "read_csv", return_value=pd.DataFrame({"channel_name": ["channel"]})),
            patch.object(main.sa, "load_channel_data", return_value=channel_data),
            patch.object(main.sa, "filter_shutters", side_effect=lambda result, _: result),
            patch.object(main.da, "save_pickle", side_effect=lambda data, name: saved.update({name: data})),
        ):
            main.main(save=True)

        self.assertIs(saved["diamon_data"], measurements)
        self.assertIs(saved["shutter_data"], channel_data)


class LongFormatFileTests(unittest.TestCase):
    def test_long_files_are_loaded_only_as_summary_data(self):
        files = ["measurement/LONG_OUT_data.txt", "measurement/LONG_rate.txt"]
        summary_data = pd.DataFrame({"value": [1]})

        with (
            patch.object(diamon.glob, "glob", return_value=files),
            patch.object(diamon, "read_data_file", return_value=summary_data) as read_data_file,
        ):
            measurement = diamon.diamon(summary=True)
            measurement.read_folder("measurement")

        self.assertEqual(read_data_file.call_count, 2)
        self.assertIs(measurement.summary_out, summary_data)
        self.assertIs(measurement.summary_rate, summary_data)
        self.assertFalse(hasattr(measurement, "out_data"))
        self.assertFalse(hasattr(measurement, "rate_data"))

    def test_long_files_are_ignored_when_summary_is_disabled(self):
        with (
            patch.object(diamon.glob, "glob", return_value=["measurement/LONG_OUT_data.txt"]),
            patch.object(diamon, "read_data_file") as read_data_file,
        ):
            measurement = diamon.diamon()
            measurement.read_folder("measurement")

        read_data_file.assert_not_called()
        self.assertFalse(hasattr(measurement, "summary_out"))


if __name__ == "__main__":
    unittest.main()
