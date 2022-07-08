from pathlib import Path

import pytest
import pandas as pd
import xarray as xr

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import (
    north_atlantic_oscillation,
    southern_annular_mode,
)
from climate_index_collection.output import compute_index


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_returntype_of_compute_index(source_name):
    """Ensure that function compute_index returns xarray DataArray."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    
    # Compute SAM index
    SAM_data_array=compute_index(data_path=TEST_DATA_PATH, data_source_name=source_name, index_function=southern_annular_mode)

    # Check, if calculated SAM index has type xarray.DataArray
    assert type(SAM_data_array) is xr.DataArray

