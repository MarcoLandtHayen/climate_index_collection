from pathlib import Path

import numpy as np
import pytest
import scipy as sp
import xarray as xr

from numpy.testing import assert_allclose, assert_almost_equal

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import (
    atlantic_multidecadal_oscillation,
    eastern_subtropical_indian_ocean_SST,
    el_nino_southern_oscillation_3,
    el_nino_southern_oscillation_4,
    el_nino_southern_oscillation_12,
    el_nino_southern_oscillation_34,
    hurricane_main_development_region_SST,
    mediterranean_SST,
    north_atlantic_oscillation,
    north_atlantic_oscillation_pc,
    north_atlantic_sea_surface_salinity,
    north_atlantic_sea_surface_salinity_east,
    north_atlantic_sea_surface_salinity_west,
    north_pacific,
    pacific_decadal_oscillation_pc,
    sahel_precipitation_anomaly,
    sea_air_surface_temperature_anomaly_north_all,
    sea_air_surface_temperature_anomaly_north_land,
    sea_air_surface_temperature_anomaly_north_ocean,
    sea_air_surface_temperature_anomaly_south_all,
    sea_air_surface_temperature_anomaly_south_land,
    sea_air_surface_temperature_anomaly_south_ocean,
    south_atlantic_sea_surface_salinity,
    southern_annular_mode,
    southern_annular_mode_pc,
    southern_oscillation,
    tropical_north_atlantic_SST,
    tropical_south_atlantic_SST,
    western_subtropical_indian_ocean_SST,
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
def test_SOI_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate Southern Oscillation index
    SOI = southern_oscillation(data_set)

    # Check, if calculated SOI only has one dimension: 'time'
    assert SOI.dims[0] == "time"
    assert len(SOI.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SOI_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate Southern Oscillation index
    SOI = southern_oscillation(data_set)

    # Check, if calculated SOI has zero mean and unit std dev:
    assert_almost_equal(actual=SOI.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=SOI.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SOI_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate Southern Oscillation index
    SOI = southern_oscillation(data_set)

    assert SOI.name == "SOI"


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
def test_enso12_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 1+2 index
    ENSO12 = el_nino_southern_oscillation_12(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert ENSO12.dims[0] == "time"
    assert len(ENSO12.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_enso12_zeromean(source_name):
    """Ensure that ENSO 1+2 has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 1+2 index
    ENSO12 = el_nino_southern_oscillation_12(data_set)

    # Check, if calculated ENSO 1+2 index has zero mean:
    assert_almost_equal(actual=ENSO12.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_ENSO12_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 1+2 index
    result = el_nino_southern_oscillation_12(data_set)

    assert result.name == "ENSO12"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_enso3_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3 index
    ENSO3 = el_nino_southern_oscillation_3(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert ENSO3.dims[0] == "time"
    assert len(ENSO3.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_enso3_zeromean(source_name):
    """Ensure that ENSO 3 has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3 index
    ENSO3 = el_nino_southern_oscillation_3(data_set)

    # Check, if calculated ENSO 3 index has zero mean:
    assert_almost_equal(actual=ENSO3.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_ENSO3_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3 index
    result = el_nino_southern_oscillation_3(data_set)

    assert result.name == "ENSO3"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_enso34_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3.4 index
    ENSO34 = el_nino_southern_oscillation_34(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert ENSO34.dims[0] == "time"
    assert len(ENSO34.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_ENSO34_zeromean(source_name):
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
def test_enso4_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 4 index
    ENSO4 = el_nino_southern_oscillation_4(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert ENSO4.dims[0] == "time"
    assert len(ENSO4.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_enso4_zeromean(source_name):
    """Ensure that ENSO 4 has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 4 index
    ENSO4 = el_nino_southern_oscillation_4(data_set)

    # Check, if calculated ENSO 4 index has zero mean:
    assert_almost_equal(actual=ENSO4.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_ENSO4_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 4 index
    result = el_nino_southern_oscillation_4(data_set)

    assert result.name == "ENSO4"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_TNA_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_TNA = tropical_north_atlantic_SST(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSTA_TNA.dims[0] == "time"
    assert len(SSTA_TNA.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_TNA_zeromean(source_name):
    """Ensure that Tropical North Atlantic SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_TNA = tropical_north_atlantic_SST(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSTA_TNA.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_TNA_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = tropical_north_atlantic_SST(data_set)

    assert result.name == "SSTA_TNA"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_TSA_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_TSA = tropical_south_atlantic_SST(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSTA_TSA.dims[0] == "time"
    assert len(SSTA_TSA.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_TSA_zeromean(source_name):
    """Ensure that Tropical South Atlantic SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_TSA = tropical_south_atlantic_SST(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSTA_TSA.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_TSA_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = tropical_south_atlantic_SST(data_set)

    assert result.name == "SSTA_TSA"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_ESIO_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_ESIO = eastern_subtropical_indian_ocean_SST(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSTA_ESIO.dims[0] == "time"
    assert len(SSTA_ESIO.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_ESIO_zeromean(source_name):
    """Ensure that Eastern Subtropical Indian Ocean SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_ESIO = eastern_subtropical_indian_ocean_SST(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSTA_ESIO.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_ESIO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = eastern_subtropical_indian_ocean_SST(data_set)

    assert result.name == "SSTA_ESIO"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_WSIO_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_WSIO = western_subtropical_indian_ocean_SST(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSTA_WSIO.dims[0] == "time"
    assert len(SSTA_WSIO.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_WSIO_zeromean(source_name):
    """Ensure that Western Subtropical Indian Ocean SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_WSIO = western_subtropical_indian_ocean_SST(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSTA_WSIO.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_WSIO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = western_subtropical_indian_ocean_SST(data_set)

    assert result.name == "SSTA_WSIO"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_MED_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_MED = mediterranean_SST(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSTA_MED.dims[0] == "time"
    assert len(SSTA_MED.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_MED_zeromean(source_name):
    """Ensure that Mediterranean Sea SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_MED = mediterranean_SST(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSTA_MED.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_MED_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = mediterranean_SST(data_set)

    assert result.name == "SSTA_MED"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_HMDR_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_HMDR = hurricane_main_development_region_SST(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSTA_HMDR.dims[0] == "time"
    assert len(SSTA_HMDR.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_HMDR_zeromean(source_name):
    """Ensure that Mediterranean Sea SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSTA_HMDR = hurricane_main_development_region_SST(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSTA_HMDR.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSTA_HMDR_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = hurricane_main_development_region_SST(data_set)

    assert result.name == "SSTA_HMDR"


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
def test_NASSS_zeromean(source_name):
    """Ensure that the index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS index
    NASSS = north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated NASSS index has zero mean:
    assert_almost_equal(actual=NASSS.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_W_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS_W index
    NASSS_W = north_atlantic_sea_surface_salinity_west(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert NASSS_W.dims[0] == "time"
    assert len(NASSS_W.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_W_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS_W index
    NASSS_W = north_atlantic_sea_surface_salinity_west(data_set)

    assert NASSS_W.name == "NASSS_W"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_W_zeromean(source_name):
    """Ensure that the index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS index
    NASSS_W = north_atlantic_sea_surface_salinity_west(data_set)

    # Check, if calculated NASSS index has zero mean:
    assert_almost_equal(actual=NASSS_W.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_E_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS_W index
    NASSS_E = north_atlantic_sea_surface_salinity_east(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert NASSS_E.dims[0] == "time"
    assert len(NASSS_E.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_E_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS_E index
    NASSS_E = north_atlantic_sea_surface_salinity_east(data_set)

    assert NASSS_E.name == "NASSS_E"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NASSS_E_zeromean(source_name):
    """Ensure that the index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS index
    NASSS_E = north_atlantic_sea_surface_salinity_east(data_set)

    # Check, if calculated NASSS index has zero mean:
    assert_almost_equal(actual=NASSS_E.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SASSS_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SASSS index
    SASSS = south_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated NASSS index only has one dimension: 'time'
    assert SASSS.dims[0] == "time"
    assert len(SASSS.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SASSS_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NASSS index
    SASSS = south_atlantic_sea_surface_salinity(data_set)

    assert SASSS.name == "SASSS"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SASSS_zeromean(source_name):
    """Ensure that the index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate SASSS index
    SASSS = south_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated NASSS index has zero mean:
    assert_almost_equal(actual=SASSS.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate AMO index
    AMO = atlantic_multidecadal_oscillation(data_set)

    # Check, if calculated AMO index only has one dimension: 'time'
    assert AMO.dims[0] == "time"
    assert len(AMO.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_zeromean(source_name):
    """Ensure that AMO has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate AMO index
    AMO = atlantic_multidecadal_oscillation(data_set)

    # Check, if calculated AMO index has zero mean:
    assert_almost_equal(actual=AMO.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NAO index
    AMO = atlantic_multidecadal_oscillation(data_set)

    assert AMO.name == "AMO"


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
def test_sahel_precip_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    Sahel_precip = sahel_precipitation_anomaly(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert Sahel_precip.dims[0] == "time"
    assert len(Sahel_precip.dims) == 1


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


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NP index
    NP = north_pacific(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert NP.dims[0] == "time"
    assert len(NP.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_zeromean(source_name):
    """Ensure that NP index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NP index
    NP = north_pacific(data_set)

    # Check, if calculated NP index has zero mean:
    assert_almost_equal(actual=NP.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate NP index
    result = north_pacific(data_set)

    assert result.name == "NP"
