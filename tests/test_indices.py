from pathlib import Path

import pytest
import xarray as xr
import numpy as np

from numpy.testing import assert_almost_equal, assert_allclose

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import (
    north_atlantic_oscillation,
    southern_annular_mode,
    el_nino_southern_oscillation_34,
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
def test_NAO_zeromean(source_name):
    """Ensure that NAO has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index
    NAO = north_atlantic_oscillation(data_set)

    # Check, if calculated NAO index has zero mean:
    assert_almost_equal(actual=NAO.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index
    NAO = north_atlantic_oscillation(data_set)

    assert NAO.name == "NAO"

@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_enso34_zeromean(source_name, rtol = 1e-8):
    """Checks if the mean of the ENSO3.4 anomalies are close to 0 using numpy.allclose()
    rtol : float 
        relative accuracy Default of 1e-6.
    
    Absolute accuracy is calculated with (max(data_set) - min(data_set)) * rtol
    For further information look at numpy.allclose()
    From numpy:
    "The tolerance values are positive, typically very small numbers. 
    The relative difference (rtol * abs(b)) and the absolute difference atol are added together 
    to compare against the absolute difference between a and b."
    """
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
    
    # Calculate ENSO 3.4 index
    result = el_nino_southern_oscillation_34(data_set=data_set)
    # to calculate the absolute tolerance, we use :
    # ( max(data_set) - min(data_set) ) * rtol
    min_value = data_set["sea-surface-temperature"].min().values
    max_value = data_set["sea-surface-temperature"].max().values
    atol = (max_value-min_value)*rtol
    result_mean = result.mean("time")
    # note that rtol has have no effect on the final values, because desired is 0
    assert_allclose(
                actual = result_mean, 
                desired = result_mean*0,
                rtol = rtol,
                atol = atol,
                equal_nan = True)

@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_ENSO34_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index
    result = el_nino_southern_oscillation_34(data_set)

    assert result.name == "ENSO34"