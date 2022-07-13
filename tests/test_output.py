from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import (
    north_atlantic_oscillation,
    southern_annular_mode,
)
from climate_index_collection.output import compute_index, index_dataarray_to_dataframe


# Test data
#
# data = [0.5, 1, -0.3, -0.7]
# dim: only 'time'
# name: 'SAM'


@pytest.fixture
def example_data_array():
    data = xr.DataArray(
        [0.5, 1, -0.3, -0.7],
        dims=("time"),
        name="SAM",
    )
    return data


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_returntype_of_compute_index(source_name):
    """Ensure that function compute_index returns xarray DataArray."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"

    # Compute SAM index
    SAM_data_array = compute_index(
        data_path=TEST_DATA_PATH,
        data_source_name=source_name,
        index_function=southern_annular_mode,
    )

    # Check, if calculated SAM index has type xarray.DataArray
    assert type(SAM_data_array) is xr.DataArray


def test_conversion_to_dataframe(example_data_array):
    """Ensure that function index_dataarray_to_dataframe correctly converts given xarray DataArray to pandas dataframe. Use default model name FOCI."""

    assert all(
        index_dataarray_to_dataframe(index_data_array=example_data_array)["time"].values
        == np.array([0, 1, 2, 3])
    )
    assert all(
        index_dataarray_to_dataframe(index_data_array=example_data_array)[
            "model"
        ].values
        == np.array(["FOCI", "FOCI", "FOCI", "FOCI"])
    )
    assert all(
        index_dataarray_to_dataframe(index_data_array=example_data_array)[
            "index"
        ].values
        == np.array(["SAM", "SAM", "SAM", "SAM"])
    )
    assert all(
        index_dataarray_to_dataframe(index_data_array=example_data_array)[
            "value"
        ].values
        == np.array([0.5, 1, -0.3, -0.7])
    )

    
@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_returntype_of_conversion_to_dataframe(source_name):
    """Ensure that function index_dataarray_to_dataframe returns pandas dataframe."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"

    # Compute SAM index
    SAM_data_array = compute_index(
        data_path=TEST_DATA_PATH,
        data_source_name=source_name,
        index_function=southern_annular_mode,
    )
    
    # Convert to pandas DataFrame
    SAM_df = index_dataarray_to_dataframe(index_data_array=SAM_data_array)

    # Check, if calculated SAM index has type pandas.DataFrame
    assert type(SAM_df) is pd.DataFrame