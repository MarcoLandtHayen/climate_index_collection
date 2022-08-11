import math
import weakref

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from numpy.testing import assert_almost_equal
from xarray import DataArray

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.reductions import (
    grouped_mean_weighted,
    monthly_anomalies_unweighted,
    monthly_mean_unweighted,
)


# ========
# CREATE TEST DATA PARAMETERS AND FUNCTIONS
# ========
@pytest.fixture
def example_data_01():
    lon = np.array([120, 140, 150])
    lat = np.array([-10, -5, 0])
    time = pd.to_datetime(["2020-02-13", "2021-06-13", "2021-08-13", "2022-02-13"])
    weights = time.days_in_month
    # create dummy dataset
    data = DataArray(
        np.array(
            [
                [[np.nan, 0.0, 57.0], [57.0, 57.0, 57.0], [0.0, 0.0, 57.0]],
                [[0.0, 0.0, 57.0], [0.0, 0.0, 0.0], [0.0, 57.0, 0.0]],
                [[0.0, 57.0, 0.0], [57.0, 0.0, 0.0], [0.0, 57.0, 57.0]],
                [[57.0, 0.0, 0.0], [57.0, 0.0, 0.0], [57.0, 57.0, 57.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return data


@pytest.fixture
def example_weights_01():
    lon = np.array([120, 140, 150])
    lat = np.array([-10, -5, 0])
    time = pd.to_datetime(["2020-02-13", "2021-06-13", "2021-08-13", "2022-02-13"])
    # create dummy dataset
    data = DataArray(
        time.days_in_month,
        dims=("time"),
        coords={"time": time},
    )
    return data


@pytest.fixture
def example_unweighted_mean_01():
    lon = np.array([120, 140, 150])
    lat = np.array([-10, -5, 0])
    time = pd.to_datetime(["2020-02-13", "2021-06-13", "2021-08-13", "2022-02-13"])
    weights = time.days_in_month
    # create dummy dataset
    data = DataArray(
        np.array(
            [
                [[57, 0.0, 28.5], [57, 28.5, 28.5], [28.5, 28.5, 57]],
                [[0.0, 0.0, 57.0], [0.0, 0.0, 0.0], [0.0, 57.0, 0.0]],
                [[0.0, 57.0, 0.0], [57.0, 0.0, 0.0], [0.0, 57.0, 57.0]],
            ]
        ),
        dims=("month", "lat", "lon"),
        coords={"month": np.unique(time.month), "lat": lat, "lon": lon},
    )
    return data


@pytest.fixture
def example_weighted_mean_01():
    lon = np.array([120, 140, 150])
    lat = np.array([-10, -5, 0])
    time = pd.to_datetime(["2020-02-13", "2021-06-13", "2021-08-13", "2022-02-13"])
    weights = time.days_in_month
    # create dummy dataset
    data = DataArray(
        np.array(
            [
                [[57.0, 0.0, 29.0], [57.0, 29.0, 29.0], [28.0, 28.0, 57.0]],
                [[0.0, 0.0, 57.0], [0.0, 0.0, 0.0], [0.0, 57.0, 0.0]],
                [[0.0, 57.0, 0.0], [57.0, 0.0, 0.0], [0.0, 57.0, 57.0]],
            ]
        ),
        dims=("month", "lat", "lon"),
        coords={"month": np.unique(time.month), "lat": lat, "lon": lon},
    )
    return data


@pytest.fixture
def example_unweighted_anomalies_01():
    lon = np.array([120, 140, 150])
    lat = np.array([-10, -5, 0])
    time = pd.to_datetime(["2020-02-13", "2021-06-13", "2021-08-13", "2022-02-13"])
    weights = time.days_in_month
    # create dummy dataset
    data = DataArray(
        np.array(
            [
                [[np.nan, 0.0, 28.5], [0.0, 28.5, 28.5], [-28.5, -28.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, -28.5], [0.0, -28.5, -28.5], [28.5, 28.5, 0.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return data


# ========
# Basic tests with the test data created above
# ========


def test_grouped_mean_weighted(
    example_data_01, example_weights_01, example_weighted_mean_01
):
    """Checks if the groupby weighting function gives proper results."""
    result = grouped_mean_weighted(
        dobj=example_data_01,
        weights=example_weights_01,
        dim="time",
        groupby_dim="time.month",
    )
    assert result.equals(example_weighted_mean_01)


def test_monthly_mean_unweighted(example_data_01, example_unweighted_mean_01):
    """Checks if the monthly mean unweighted function gives proper results."""
    result = monthly_mean_unweighted(dobj=example_data_01)
    assert result.equals(example_unweighted_mean_01)


def test_monthly_anomalies_unweighted(example_data_01, example_unweighted_anomalies_01):
    """Checks if the monthly anomalies unweighted function gives proper results."""
    result = monthly_anomalies_unweighted(dobj=example_data_01)
    assert result.equals(example_unweighted_anomalies_01)


# ========
# Test if the means created by the anomalies functions are close to zero.
# ========
@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_monthly_anomalies_unweighted_zeromean(source_name, relative_tolerance=1e-5):
    """Checks if the mean of the monthly anomalies unweighted are all close to 0.
    The test will be performed for one spatial gridpoint near 40N and 30W.
    Variables used are
    - sea-surface-temperature
    - sea-level-pressure
    - geopotential-height

    Parameters
    ----------
    source_name: str or DataSet
        Test dataset name.
        Or DataSet directly
    relative_tolerance : float
        Relative tolerance which shall be used to derive the decimal accuracy
        for each variable individually.
        Default to 1e-5.

    """
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
    data_set = data_set.sel(lat=40, method="nearest").sel(lon=330, method="nearest")
    # For the three variables check if the mean of the anomalies is 0
    for variable in [
        "sea-surface-temperature",
        "sea-level-pressure",
        "geopotential-height",
    ]:
        current_data_array = data_set[variable]
        anomalies = monthly_anomalies_unweighted(dobj=current_data_array)
        anomalies_mean = anomalies.mean("time")

        min_value = current_data_array.min().values
        max_value = current_data_array.max().values
        absolute_tolerance = (max_value - min_value) * relative_tolerance
        # calculate the desired decimal accuracy as
        # -1 * Order(absolute accuracy)
        decimal = -1 * math.floor(math.log(absolute_tolerance, 10))
        assert_almost_equal(actual=anomalies_mean.values, desired=0, decimal=decimal)
