import pytest
import xarray as xr
from numpy.testing import assert_almost_equal
from pathlib import Path
from climate_index_collection.indices import southern_annular_mode
from climate_index_collection.data_loading import load_data_set


@pytest.mark.parametrize("source_name", ["FOCI", "CESM"])
def test_SAM_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load FOCI test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SAM index
    SAM = southern_annular_mode(data_set)

    # Check, if calculated SAM index only has one dimension: 'time'
    assert SAM.dims[0] == 'time'
    assert len(SAM.dims) == 1
    

@pytest.mark.parametrize("source_name", ["FOCI", "CESM"])
def test_SAM_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load FOCI test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SAM index
    SAM = southern_annular_mode(data_set)

    # Check, if calculated SAM index has zero mean and unit std dev:
    assert_almost_equal(actual=SAM.mean('time').values[()], desired=0, decimal=3)
    assert_almost_equal(actual=SAM.std('time').values[()], desired=1, decimal=3)
    