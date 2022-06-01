import xarray as xr
from numpy.testing import assert_almost_equal
from pathlib import Path
from climate_index_collection.indices import southern_annular_mode
from climate_index_collection.data_loading import load_data_set


def test_SAM_metadata():
    """Ensure that index only contains time dimension."""
    # Load FOCI test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    foci_ds = load_data_set(data_path=TEST_DATA_PATH, data_source_name="FOCI")

    # Calculate SAM index
    foci_SAM = southern_annular_mode(foci_ds, slp_name="slp")

    # Check, if calculated SAM index only has one dimension: 'time'
    assert foci_SAM.dims[0] == 'time'
    assert len(foci_SAM.dims) == 1
    
    
def test_SAM_standardisation():
    """Ensure that standardisation works correctly."""
    # Load FOCI test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    foci_ds = load_data_set(data_path=TEST_DATA_PATH, data_source_name="FOCI")
    
    # Calculate SAM index
    foci_SAM = southern_annular_mode(foci_ds, slp_name="slp")

    # Check, if calculated SAM index has zero mean and unit std dev:
    assert_almost_equal(actual=foci_SAM.mean('time').values[()], desired=0, decimal=3)
    assert_almost_equal(actual=foci_SAM.std('time').values[()], desired=1, decimal=3)
    