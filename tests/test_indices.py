from pathlib import Path

import numpy as np
import pytest
import scipy as sp
import xarray as xr

from numpy.testing import assert_allclose, assert_almost_equal

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import (
    el_nino_southern_oscillation_34,
    north_atlantic_oscillation,
    north_atlantic_oscillation_pc,
    north_atlantic_sea_surface_salinity,
    pacific_decadal_oscillation_pc,
    sahel_precipitation_anomaly,
    sea_air_surface_temperature_anomaly_north_all,
    sea_air_surface_temperature_anomaly_north_land,
    sea_air_surface_temperature_anomaly_north_ocean,
    sea_air_surface_temperature_anomaly_south_all,
    sea_air_surface_temperature_anomaly_south_land,
    sea_air_surface_temperature_anomaly_south_ocean,
    southern_annular_mode,
    southern_annular_mode_pc,
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
def test_SAM_PC_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SAM index
    SAM = southern_annular_mode_pc(data_set)

    # Check, if calculated SAM index only has one dimension: 'time'
    assert SAM.dims[0] == "time"
    assert len(SAM.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_PC_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SAM index
    SAM = southern_annular_mode_pc(data_set)

    # Check, if calculated SAM index has zero mean and unit std dev:
    assert_almost_equal(actual=SAM.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=SAM.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_PC_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SAM index
    SAM = southern_annular_mode_pc(data_set)

    assert SAM.name == "SAM_PC"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_PC_correlation(source_name):
    """Ensure that PC-based index is positively correlated to regular SAM index."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate regular and PC-based SAM index
    SAM = southern_annular_mode(data_set)
    SAM_PC = southern_annular_mode_pc(data_set)

    assert np.corrcoef(np.stack([SAM.values, SAM_PC.values]))[0, 1] > 0


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
def test_NAO_PC_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index PC-based
    NAO_PC = north_atlantic_oscillation_pc(data_set)

    # Check, if calculated NAO index only has one dimension: 'time'
    assert NAO_PC.dims[0] == "time"
    assert len(NAO_PC.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_PC_standardisationn(source_name):
    """Ensure that NAO has zero mean and unit std dev."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index PC-based
    NAO_PC = north_atlantic_oscillation_pc(data_set)

    # Check, if calculated NAO index has zero mean:
    assert_almost_equal(actual=NAO_PC.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=NAO_PC.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_PC_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index PC-based
    NAO_PC = north_atlantic_oscillation_pc(data_set)

    assert NAO_PC.name == "NAO_PC"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_PC_correlation(source_name):
    """Ensure that PC-based index is positively correlated to station based NAO index."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate station based and PC-based NAO index
    NAO = north_atlantic_oscillation(data_set)
    NAO_PC = north_atlantic_oscillation_pc(data_set)

    assert np.corrcoef(np.stack([NAO.values, NAO_PC.values]))[0, 1] > 0


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_enso34_zeromean(source_name):
    """Ensure that ENSO 3.4 has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3.4 index
    ENSO34 = el_nino_southern_oscillation_34(data_set)

    # Check, if calculated ENSO 3.4 index has zero mean:
    assert_almost_equal(actual=ENSO34.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_ENSO34_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3.4 index
    result = el_nino_southern_oscillation_34(data_set)

    assert result.name == "ENSO34"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS index
    NASSS = north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated NASSS index only has one dimension: 'time'
    assert NASSS.dims[0] == "time"
    assert len(NASSS.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS index
    NASSS = north_atlantic_sea_surface_salinity(data_set)

    assert NASSS.name == "NASSS"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS index
    NASSS = north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated NASSS index has zero mean and unit std dev:
    assert_almost_equal(actual=NASSS.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=NASSS.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
@pytest.mark.parametrize(
    "index_function",
    [
        sea_air_surface_temperature_anomaly_north_all,
        sea_air_surface_temperature_anomaly_north_land,
        sea_air_surface_temperature_anomaly_north_ocean,
        sea_air_surface_temperature_anomaly_south_all,
        sea_air_surface_temperature_anomaly_south_land,
        sea_air_surface_temperature_anomaly_south_ocean,
    ],
)
def test_SASTAI_metadata(source_name, index_function):
    """Ensure that index of the SASTAI index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SASTAI
    SASTAI = index_function(data_set)

    # Check, if calculated SASTAI only has one dimension: 'time'
    assert SASTAI.dims[0] == "time"
    assert len(SASTAI.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
@pytest.mark.parametrize(
    "index_function",
    [
        sea_air_surface_temperature_anomaly_north_all,
        sea_air_surface_temperature_anomaly_north_land,
        sea_air_surface_temperature_anomaly_north_ocean,
        sea_air_surface_temperature_anomaly_south_all,
        sea_air_surface_temperature_anomaly_south_land,
        sea_air_surface_temperature_anomaly_south_ocean,
    ],
)
def test_SASTAI_zeromean(source_name, index_function):
    """Ensure that mean of the index is zero."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SASTAI
    SASTAI = index_function(data_set)
    # Check, if calculated SASTAI has zero mean:
    assert_almost_equal(actual=SASTAI.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
@pytest.mark.parametrize(
    "index_function",
    [
        sea_air_surface_temperature_anomaly_north_all,
        sea_air_surface_temperature_anomaly_north_land,
        sea_air_surface_temperature_anomaly_north_ocean,
        sea_air_surface_temperature_anomaly_south_all,
        sea_air_surface_temperature_anomaly_south_land,
        sea_air_surface_temperature_anomaly_south_ocean,
    ],
)
def test_SASTAI_north_all_naming(source_name, index_function):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SASTAI
    SASTAI = index_function(data_set)

    assert SASTAI.name.startswith("SASTAI-")


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_sahel_precip_zeromean(source_name):
    """Ensure that Sahel precipitation anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate Sahel precipitation anomaly index
    Sahel_precip = sahel_precipitation_anomaly(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(
        actual=Sahel_precip.mean("time").values[()], desired=0, decimal=3
    )


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_sahel_precip_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate Sahel precipitation anomaly index index
    result = sahel_precipitation_anomaly(data_set)

    assert result.name == "SPAI"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PDO_PC_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate PDO index PC-based
    PDO_PC = pacific_decadal_oscillation_pc(data_set)

    # Check, if calculated PDO index only has one dimension: 'time'
    assert PDO_PC.dims[0] == "time"
    assert len(PDO_PC.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PDO_PC_standardisationn(source_name):
    """Ensure that PDO has zero mean and unit std dev."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index PC-based
    PDO_PC = pacific_decadal_oscillation_pc(data_set)

    # Check, if calculated PDO index has zero mean:
    assert_almost_equal(actual=PDO_PC.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=PDO_PC.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PDO_PC_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate PDO index PC-based
    PDO_PC = pacific_decadal_oscillation_pc(data_set)

    assert PDO_PC.name == "PDO_PC"
