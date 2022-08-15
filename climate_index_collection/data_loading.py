from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .timestamp_handling import fix_monthly_time_stamps


def find_data_files(data_path="data/test_data/", data_source_name="FOCI"):
    """Find all files for given data source.

    Parameters
    ----------
    data_path: str | path
        Location of the data files. Defaults to "data/test_data/".
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".

    Returns
    -------
    list
        Paths to all data files.

    """
    data_files = list(sorted(Path(data_path).glob(f"{data_source_name}/*.nc")))
    return data_files


def load_and_preprocess_single_data_file(file_name, **kwargs):
    """Load and preprocess individual data file.

    Current pre-processing steps:
    - squeeze (to get rid of, e.g., some of the vertical degenerate dims)
    - fixing monthly timestamps

    Parameters
    ----------
    file_name: str or pathlike
        File name.

    All kwargs will be passed on to xarray.open_dataset().

    Returns
    -------
    xarray.Dataset
        Dataset.

    """
    ds = xr.open_dataset(file_name, **kwargs)

    # get rid of singleton dims
    ds = ds.squeeze()

    # fix time stamps to always be mid month
    ds = fix_monthly_time_stamps(ds)

    return ds


def load_and_preprocess_multiple_data_files(file_names, **kwargs):
    """Loads and preprocesses multiple files.

    Parameters
    ---------
    file_names: iterable
        File names (str or pathlike).

    All kwargs will be passed on to load_and_preprocess_single_data_file().

    Returns
    -------
    xarray.Dataset
        Dataset.

    """
    return xr.merge(
        (load_and_preprocess_single_data_file(fname, **kwargs) for fname in file_names)
    )


def load_data_set(data_path="data/test_data/", data_source_name="FOCI", **kwargs):
    """Load dataset.

    Parameters
    ----------
    data_path: str | path
        Location of the data files. Defaults to "data/test_data/".
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".

    All kwargs will be passed to xarray.open_mfdataset. Use this for, e.g., chunking.

    Returns
    -------
    xarray.Dataset
        Multifile dataset with (pointers to) all data.

    """
    data_files = find_data_files(data_path=data_path, data_source_name=data_source_name)
    raw_data_set = load_and_preprocess_multiple_data_files(data_files, **kwargs)
    data_set = standardize_metadata(raw_data_set, data_source_name=data_source_name)

    return data_set


# this is dangerous...
VARNAME_MAPPING = {
    "FOCI": {
        "slp": "sea-level-pressure",
        "tsw": "sea-surface-temperature",
        "geopoth": "geopotential-height",
        "temp2": "sea-air-temperature",
        "sosaline": "sea-surface-salinity",
        "precip": "precipitation",
    },
    "CESM": {
        "PSL": "sea-level-pressure",
        "SST": "sea-surface-temperature",
        "Z3": "geopotential-height",
        "TS": "sea-air-temperature",
        "SALT": "sea-surface-salinity",
        "PRECT": "precipitation",
    },
}


def standardize_metadata(raw_data_set=None, data_source_name="FOCI"):
    """Standardize metadata (dims, coords, attributes, varnames).

    Parameters
    ----------
    raw_data_set: xarray.Dataset
        Dataset with potentially non-standard metadata.
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".

    Returns
    xarray.Dataset
        Dataset with standardised metadata.

    """

    data_set = raw_data_set.rename_vars(VARNAME_MAPPING[data_source_name])
    return data_set
