# Tech Preamble
import xarray as xr
from pathlib import Path
from climate_index_collection.indices import southern_annular_mode

# Load FOCI test data
data_path = "../data/test_data/"
foci_data_files = list(sorted(Path(data_path).glob("FOCI/*.nc")))
foci_ds = xr.open_mfdataset(foci_data_files)

# Calculate SAM index
foci_SAM = southern_annular_mode(foci_ds, slp_name="slp")

def test_SAM():
    # Check, if calculated SAM index only has one dimension: 'time'
    assert (foci_SAM.dims[0] == 'time') & len(foci_SAM.dims) == 1
    # Check, if calculated SAM index has zero mean and unit std dev:
    assert (foci_SAM.mean('time').values == 0) & (foci_SAM.std('time').values == 1)