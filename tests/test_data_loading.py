from datetime import datetime
from pathlib import Path

import cftime
import numpy as np
import pytest

from climate_index_collection.data_loading import (
    OCEAN_ONLY_VARS,
    VARNAME_MAPPING,
    load_data_set,
)


TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"


@pytest.mark.parametrize("source_name", ["FOCI", "CESM"])
def test_loading_just_open(source_name):
    """Just load the data and do nothting."""
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)


@pytest.mark.parametrize("source_name", ["FOCI", "CESM"])
def test_loading_standardization_renamed_varnames_absent(source_name):
    """Make sure that none of the vars we map to standard names are present."""
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
    assert all(
        vn not in data_set.data_vars.keys()
        for vn in VARNAME_MAPPING[source_name].keys()
    )


@pytest.mark.parametrize("source_name", ["FOCI", "CESM"])
def test_missing_data_constant_in_time(source_name):
    """Make sure validity of all grid points is constant in time."""
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
    for dv in data_set.data_vars.values():
        if "time" in dv.dims:
            time_variability_of_masking = dv.isnull().std("time")
            np.testing.assert_almost_equal(
                time_variability_of_masking.data, desired=0, decimal=5
            )


@pytest.mark.parametrize("source_name", ["FOCI", "CESM"])
def test_missing_data_only_in_desired_vars(source_name):
    """Make sure only ocean-only vars are masked."""
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
    for varname, var in data_set.data_vars.items():
        if varname in OCEAN_ONLY_VARS:
            assert 0 < var.isnull().sum().data[()]
        else:
            assert 0 == var.isnull().sum().data[()]


@pytest.mark.parametrize("source_name", ["FOCI", "CESM"])
def test_ensure_there_is_ocean_mask(source_name):
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
    assert "is_over_ocean" in data_set.data_vars
    assert np.issubdtype(data_set["is_over_ocean"].dtype, np.bool_)
