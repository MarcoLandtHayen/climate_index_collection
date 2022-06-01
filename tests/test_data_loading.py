from pathlib import Path

import pytest

from climate_index_collection.data_loading import load_data_set


TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"


@pytest.mark.parametrize("source_name", ["FOCI", "CESM"])
def test_loading(source_name):
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
