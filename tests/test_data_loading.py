from pathlib import Path

import pytest

from climate_index_collection.data_loading import (
    load_data_set,
    VARNAME_MAPPING,
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
