from pathlib import Path

import pytest
import xarray as xr

from numpy.testing import assert_almost_equal

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import (
    north_atlantic_oscillation,
    southern_annular_mode,
)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SAM index
    SAM = southern_annular_mode(data_set)

    # Check, if calculated SAM index only has one dimension: 'time'
    assert SAM.dims[0] == "time"
    assert len(SAM.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SAM index
    SAM = southern_annular_mode(data_set)

    # Check, if calculated SAM index has zero mean and unit std dev:
    assert_almost_equal(actual=SAM.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=SAM.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SAM index
    SAM = southern_annular_mode(data_set)

    assert SAM.name == "SAM"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index
    NAO = north_atlantic_oscillation(data_set)

    # Check, if calculated NAO index only has one dimension: 'time'
    assert NAO.dims[0] == "time"
    assert len(NAO.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index
    NAO = north_atlantic_oscillation(data_set)

    assert NAO.name == "NAO"
