from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import (
    north_atlantic_oscillation_station,
    southern_annular_mode_zonal_mean,
)
from climate_index_collection.output import (
    compute_index,
    concat_indices,
    index_dataarray_to_dataframe,
    run_full_workflow_csv,
)


# Dummy example data
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
    # Path to test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"

    # Compute SAM index
    SAM_data_array = compute_index(
        data_path=TEST_DATA_PATH,
        data_source_name=source_name,
        index_function=southern_annular_mode_zonal_mean,
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
    np.testing.assert_allclose(
        index_dataarray_to_dataframe(index_data_array=example_data_array)[
            "value"
        ].values,
        np.array([0.5, 1, -0.3, -0.7]),
    )


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_returntype_of_conversion_to_dataframe(source_name):
    """Ensure that function index_dataarray_to_dataframe returns pandas dataframe."""
    # Path to test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"

    # Compute SAM index
    SAM_data_array = compute_index(
        data_path=TEST_DATA_PATH,
        data_source_name=source_name,
        index_function=southern_annular_mode_zonal_mean,
    )

    # Convert to pandas DataFrame
    SAM_df = index_dataarray_to_dataframe(index_data_array=SAM_data_array)

    # Check, if calculated SAM index has type pandas.DataFrame
    assert type(SAM_df) is pd.DataFrame


def test_concat_indices():
    """Ensure that resulting dataframe has unique index and does not contain duplicate rows."""
    # Path to test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"

    # Compute indices from sources and concatenate resulting dataframes
    df = concat_indices(
        data_path=TEST_DATA_PATH,
        data_source_names=["FOCI", "CESM"],
        index_functions=[
            southern_annular_mode_zonal_mean,
            north_atlantic_oscillation_station,
        ],
    )

    # Check, if resulting dataframe has unique index
    assert np.max(df.index.value_counts()) == 1

    # Check, if resulting dataframe has NO duplicate rows
    assert not (any(df.duplicated()))


def test_run_full_workflow_csv(tmp_path):
    """Ensure that output is written as csv file."""
    # Path to test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"

    # Path to output data
    outfile = tmp_path / "climate_indices.csv"

    # Compute indices from sources, concatenate resulting dataframes and output csv file
    run_full_workflow_csv(
        data_path=TEST_DATA_PATH,
        file_name=outfile,
        data_source_names=["FOCI", "CESM"],
        index_functions=[
            southern_annular_mode_zonal_mean,
            north_atlantic_oscillation_station,
        ],
    )

    # Check, if csv output file exists
    assert outfile.exists()
