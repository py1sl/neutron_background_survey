"""Stub the private influx_data_utils package when it is not installed (e.g. in CI)."""
import sys
from unittest import mock

try:
    import influx_data_utils  # noqa: F401
except ImportError:
    for name in (
        "influx_data_utils",
        "influx_data_utils.process_influx_data",
        "influx_data_utils.process_influx_data.datetime_localiser",
        "influx_data_utils.process_influx_data.cycle_datetime_formatter",
        "influx_data_utils.process_influx_data.influx_data_processor",
        "influx_data_utils.process_influx_data.influx_querier",
    ):
        sys.modules[name] = mock.MagicMock()
