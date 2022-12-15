from pathlib import Path

import cftime
import numpy as np
import pytest
import scipy as sp
import xarray as xr

from numpy.testing import assert_allclose, assert_almost_equal

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.indices import (
    atlantic_multidecadal_oscillation,
    eastern_north_atlantic_sea_surface_salinity,
    eastern_subtropical_indian_ocean_sea_surface_temperature,
    el_nino_southern_oscillation_3,
    el_nino_southern_oscillation_4,
    el_nino_southern_oscillation_12,
    el_nino_southern_oscillation_34,
    hurricane_main_development_region_sea_surface_temperature,
    mediterranean_sea_surface_temperature,
    north_atlantic_oscillation_pc,
    north_atlantic_oscillation_station,
    north_atlantic_sea_surface_salinity,
    north_pacific,
    pacific_decadal_oscillation_pc,
    sahel_precipitation,
    surface_air_temperature_north_all,
    surface_air_temperature_north_land,
    surface_air_temperature_north_ocean,
    surface_air_temperature_south_all,
    surface_air_temperature_south_land,
    surface_air_temperature_south_ocean,
    south_atlantic_sea_surface_salinity,
    southern_annular_mode_pc,
    southern_annular_mode_zonal_mean,
    southern_oscillation,
    tropical_north_atlantic_sea_surface_temperature,
    tropical_south_atlantic_sea_surface_temperature,
    western_north_atlantic_sea_surface_salinity,
    western_subtropical_indian_ocean_sea_surface_temperature,
)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_ZM_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_ZM = southern_annular_mode_zonal_mean(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SAM_ZM.dims[0] == "time"
    assert len(SAM_ZM.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_ZM_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_ZM = southern_annular_mode_zonal_mean(data_set)

    # Check, if calculated index has zero mean and unit std dev:
    assert_almost_equal(actual=SAM_ZM.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=SAM_ZM.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_ZM_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_ZM = southern_annular_mode_zonal_mean(data_set)

    assert SAM_ZM.name == "SAM_ZM"
    assert SAM_ZM.long_name == "southern_annular_mode_zonal_mean"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_PC_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_PC = southern_annular_mode_pc(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SAM_PC.dims[0] == "time"
    assert len(SAM_PC.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_PC_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_PC = southern_annular_mode_pc(data_set)

    # Check, if calculated index has zero mean and unit std dev:
    assert_almost_equal(actual=SAM_PC.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=SAM_PC.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_PC_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_PC = southern_annular_mode_pc(data_set)

    assert SAM_PC.name == "SAM_PC"
    assert SAM_PC.long_name == "southern_annular_mode_pc"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_PC_correlation(source_name):
    """Ensure that PC-based index is positively correlated to regular SAM index."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate regular and PC-based SAM index
    SAM_ZM = southern_annular_mode_zonal_mean(data_set)
    SAM_PC = southern_annular_mode_pc(data_set)

    assert np.corrcoef(np.stack([SAM_ZM.values, SAM_PC.values]))[0, 1] > 0


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SOI_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SOI = southern_oscillation(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SOI.dims[0] == "time"
    assert len(SOI.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SOI_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SOI = southern_oscillation(data_set)

    # Check, if calculated index has zero mean and unit std dev:
    assert_almost_equal(actual=SOI.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=SOI.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SOI_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SOI = southern_oscillation(data_set)

    assert SOI.name == "SOI"
    assert SOI.long_name == "southern_oscillation"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_ST_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_ST = north_atlantic_oscillation_station(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert NAO_ST.dims[0] == "time"
    assert len(NAO_ST.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_ST_zeromean(source_name):
    """Ensure that index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_ST = north_atlantic_oscillation_station(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=NAO_ST.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_ST_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_ST = north_atlantic_oscillation_station(data_set)

    assert NAO_ST.name == "NAO_ST"
    assert NAO_ST.long_name == "north_atlantic_oscillation_station"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_PC_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_PC = north_atlantic_oscillation_pc(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert NAO_PC.dims[0] == "time"
    assert len(NAO_PC.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_PC_standardisationn(source_name):
    """Ensure that index has zero mean and unit std dev."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_PC = north_atlantic_oscillation_pc(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=NAO_PC.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=NAO_PC.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_PC_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_PC = north_atlantic_oscillation_pc(data_set)

    assert NAO_PC.name == "NAO_PC"
    assert NAO_PC.long_name == "north_atlantic_oscillation_pc"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_PC_correlation(source_name):
    """Ensure that PC-based index is positively correlated to station based NAO index."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate station based and PC-based NAO index
    NAO_ST = north_atlantic_oscillation_station(data_set)
    NAO_PC = north_atlantic_oscillation_pc(data_set)

    assert np.corrcoef(np.stack([NAO_ST.values, NAO_PC.values]))[0, 1] > 0


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

    assert result.name == "ENSO_12"
    assert result.long_name == "el_nino_southern_oscillation_12"


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

    assert result.name == "ENSO_3"
    assert result.long_name == "el_nino_southern_oscillation_3"


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

    assert result.name == "ENSO_34"
    assert result.long_name == "el_nino_southern_oscillation_34"


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

    assert result.name == "ENSO_4"
    assert result.long_name == "el_nino_southern_oscillation_4"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_TNA_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_TNA = tropical_north_atlantic_sea_surface_temperature(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SST_TNA.dims[0] == "time"
    assert len(SST_TNA.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_TNA_zeromean(source_name):
    """Ensure that Tropical North Atlantic SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_TNA = tropical_north_atlantic_sea_surface_temperature(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SST_TNA.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_TNA_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = tropical_north_atlantic_sea_surface_temperature(data_set)

    assert result.name == "SST_TNA"
    assert result.long_name == "tropical_north_atlantic_sea_surface_temperature"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_TSA_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_TSA = tropical_south_atlantic_sea_surface_temperature(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SST_TSA.dims[0] == "time"
    assert len(SST_TSA.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_TSA_zeromean(source_name):
    """Ensure that Tropical South Atlantic SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_TSA = tropical_south_atlantic_sea_surface_temperature(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SST_TSA.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_TSA_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = tropical_south_atlantic_sea_surface_temperature(data_set)

    assert result.name == "SST_TSA"
    assert result.long_name == "tropical_south_atlantic_sea_surface_temperature"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_ESIO_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_ESIO = eastern_subtropical_indian_ocean_sea_surface_temperature(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SST_ESIO.dims[0] == "time"
    assert len(SST_ESIO.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_ESIO_zeromean(source_name):
    """Ensure that Eastern Subtropical Indian Ocean SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_ESIO = eastern_subtropical_indian_ocean_sea_surface_temperature(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SST_ESIO.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_ESIO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = eastern_subtropical_indian_ocean_sea_surface_temperature(data_set)

    assert result.name == "SST_ESIO"
    assert (
        result.long_name == "eastern_subtropical_indian_ocean_sea_surface_temperature"
    )


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_WSIO_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_WSIO = western_subtropical_indian_ocean_sea_surface_temperature(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SST_WSIO.dims[0] == "time"
    assert len(SST_WSIO.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_WSIO_zeromean(source_name):
    """Ensure that Western Subtropical Indian Ocean SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_WSIO = western_subtropical_indian_ocean_sea_surface_temperature(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SST_WSIO.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_WSIO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = western_subtropical_indian_ocean_sea_surface_temperature(data_set)

    assert result.name == "SST_WSIO"
    assert (
        result.long_name == "western_subtropical_indian_ocean_sea_surface_temperature"
    )


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_MED_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_MED = mediterranean_sea_surface_temperature(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SST_MED.dims[0] == "time"
    assert len(SST_MED.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_MED_zeromean(source_name):
    """Ensure that Mediterranean Sea SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_MED = mediterranean_sea_surface_temperature(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SST_MED.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_MED_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = mediterranean_sea_surface_temperature(data_set)

    assert result.name == "SST_MED"
    assert result.long_name == "mediterranean_sea_surface_temperature"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_HMDR_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_HMDR = hurricane_main_development_region_sea_surface_temperature(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SST_HMDR.dims[0] == "time"
    assert len(SST_HMDR.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_HMDR_zeromean(source_name):
    """Ensure that Mediterranean Sea SST anomaly index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SST_HMDR = hurricane_main_development_region_sea_surface_temperature(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SST_HMDR.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SST_HMDR_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = hurricane_main_development_region_sea_surface_temperature(data_set)

    assert result.name == "SST_HMDR"
    assert (
        result.long_name == "hurricane_main_development_region_sea_surface_temperature"
    )


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_NA_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_NA = north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSS_NA.dims[0] == "time"
    assert len(SSS_NA.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_NA_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_NA = north_atlantic_sea_surface_salinity(data_set)

    assert SSS_NA.name == "SSS_NA"
    assert SSS_NA.long_name == "north_atlantic_sea_surface_salinity"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_NA_zeromean(source_name):
    """Ensure that the index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_NA = north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated NASSS index has zero mean:
    assert_almost_equal(actual=SSS_NA.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_WNA_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_WNA = western_north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSS_WNA.dims[0] == "time"
    assert len(SSS_WNA.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_WNA_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_WNA = western_north_atlantic_sea_surface_salinity(data_set)

    assert SSS_WNA.name == "SSS_WNA"
    assert SSS_WNA.long_name == "western_north_atlantic_sea_surface_salinity"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_WNA_zeromean(source_name):
    """Ensure that the index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_WNA = western_north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSS_WNA.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_ENA_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_ENA = eastern_north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSS_ENA.dims[0] == "time"
    assert len(SSS_ENA.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_ENA_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_ENA = eastern_north_atlantic_sea_surface_salinity(data_set)

    assert SSS_ENA.name == "SSS_ENA"
    assert SSS_ENA.long_name == "eastern_north_atlantic_sea_surface_salinity"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_ENA_zeromean(source_name):
    """Ensure that the index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_ENA = eastern_north_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSS_ENA.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_SA_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_SA = south_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SSS_SA.dims[0] == "time"
    assert len(SSS_SA.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_SA_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_SA = south_atlantic_sea_surface_salinity(data_set)

    assert SSS_SA.name == "SSS_SA"
    assert SSS_SA.long_name == "south_atlantic_sea_surface_salinity"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SSS_SA_zeromean(source_name):
    """Ensure that the index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SSS_SA = south_atlantic_sea_surface_salinity(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SSS_SA.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    AMO = atlantic_multidecadal_oscillation(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert AMO.dims[0] == "time"
    assert len(AMO.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_zeromean(source_name):
    """Ensure that AMO has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    AMO = atlantic_multidecadal_oscillation(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=AMO.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    AMO = atlantic_multidecadal_oscillation(data_set)

    assert AMO.name == "AMO"
    assert AMO.long_name == "atlantic_multidecadal_oscillation"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
@pytest.mark.parametrize(
    "index_function",
    [
        surface_air_temperature_north_all,
        surface_air_temperature_north_land,
        surface_air_temperature_north_ocean,
        surface_air_temperature_south_all,
        surface_air_temperature_south_land,
        surface_air_temperature_south_ocean,
    ],
)
def test_SAT_metadata(source_name, index_function):
    """Ensure that index of the index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAT = index_function(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert SAT.dims[0] == "time"
    assert len(SAT.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
@pytest.mark.parametrize(
    "index_function",
    [
        surface_air_temperature_north_all,
        surface_air_temperature_north_land,
        surface_air_temperature_north_ocean,
        surface_air_temperature_south_all,
        surface_air_temperature_south_land,
        surface_air_temperature_south_ocean,
    ],
)
def test_SAT_zeromean(source_name, index_function):
    """Ensure that mean of the index is zero."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAT = index_function(data_set)
    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=SAT.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
@pytest.mark.parametrize(
    "index_function",
    [
        surface_air_temperature_north_all,
        surface_air_temperature_north_land,
        surface_air_temperature_north_ocean,
        surface_air_temperature_south_all,
        surface_air_temperature_south_land,
        surface_air_temperature_south_ocean,
    ],
)
def test_SAT_naming(source_name, index_function):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAT = index_function(data_set)

    assert SAT.name.startswith("SAT")
    assert SAT.long_name.startswith("surface_air_temperature")


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PREC_SAHEL_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    PREC_SAHEL = sahel_precipitation(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert PREC_SAHEL.dims[0] == "time"
    assert len(PREC_SAHEL.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PREC_SAHEL_zeromean(source_name):
    """Ensure that index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate Sahel precipitation anomaly index
    PREC_SAHEL = sahel_precipitation(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=PREC_SAHEL.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PREC_SAHEL_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index index
    result = sahel_precipitation(data_set)

    assert result.name == "PREC_SAHEL"
    assert result.long_name == "sahel_precipitation"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PDO_PC_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate PDO index PC-based
    PDO_PC = pacific_decadal_oscillation_pc(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert PDO_PC.dims[0] == "time"
    assert len(PDO_PC.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PDO_PC_standardisationn(source_name):
    """Ensure that index has zero mean and unit std dev."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    PDO_PC = pacific_decadal_oscillation_pc(data_set)

    # Check, if calculated index has zero mean:
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
    assert PDO_PC.long_name == "pacific_decadal_oscillation_pc"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NP = north_pacific(data_set)

    # Check, if calculated index only has one dimension: 'time'
    assert NP.dims[0] == "time"
    assert len(NP.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_zeromean(source_name):
    """Ensure that index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NP = north_pacific(data_set)

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=NP.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = north_pacific(data_set)

    assert result.name == "NP"
    assert result.long_name == "north_pacific"
