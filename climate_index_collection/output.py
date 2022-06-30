from pathlib import Path

import pandas as pd
import xarray as xr

from climate_index_collection.data_loading import load_data_set
from climate_index_collection.indices import southern_annular_mode


def compute_index(
    data_path="../data/test_data/",
    data_source_name="FOCI",
    index_function=southern_annular_mode,
):
    """Compute index from data source and return xarray DataArray.

    Parameters
    ----------
    data_path: str | path
        Location of the data files. Defaults to "data/test_data/".
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".
    index_function: function
        Index function. Defaults to southern_annular_mode.

    Returns
    -------
    xarray DataArray

    """
    data_set = load_data_set(data_path=data_path, data_source_name=data_source_name)
    index_data_array = index_function(data_set)

    return index_data_array


def index_dataarray_to_dataframe(index_data_array=None, data_source_name="FOCI"):
    """Convert index from xarray DataArray to Pandas dataframe.

    Parameters
    ----------
    index_data_array: xarray DataArray
        Defaults to None.
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".

    Returns
    -------
    Pandas dataframe

    """
    index_df = index_data_array.to_dataframe().reset_index()
    index_df["model"] = data_source_name
    index_df["index"] = index_data_array.name
    index_df = index_df.rename(columns={index_data_array.name: "value"})
    index_df = index_df.reindex(columns=["time", "model", "index", "value"])

    return index_df


def concat_indices(
    data_path="../data/test_data/", data_source_names=None, index_functions=None
):
    """Compute indices from sources and concatenate resulting dataframes.

    Parameters
    ----------
    data_path: str | path
        Location of the data files. Defaults to "data/test_data/".
    data_source_names: list of strings
        List of model names. Defaults to None.
    index_functions: list of strings
        List of indices to be computed. Defaults to None.

    Returns
    -------
    Pandas dataframe in tidy format

    """
    df_list = [
        index_dataarray_to_dataframe(
            index_data_array=compute_index(
                data_path=data_path,
                data_source_name=data_source_name,
                index_function=index_function,
            ),
            data_source_name=data_source_name,
        )
        for data_source_name in data_source_names
        for index_function in index_functions
    ]

    return pd.concat(df_list, axis=0, ignore_index=True)


def run_full_workflow_csv(
    data_path="../data/test_data/",
    file_name="output.csv",
    data_source_names=None,
    index_functions=None,
):
    """Compute indices from sources, concatenate resulting dataframes in tidy format and output as csv.

    Parameters
    ----------
    data_path: str | path
        Location of the data files. Defaults to "data/test_data/".
    file_name: string
        File name for desired output. Default: output.csv
    data_source_names: list of strings
        List of model names. Defaults to None.
    index_functions: list of strings
        List of indices to be computed. Defaults to None.

    Returns
    -------
    No return value.

    """
    # ensure output dir exists
    output_dir = Path(file_name).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = concat_indices(
        data_path=data_path,
        data_source_names=data_source_names,
        index_functions=index_functions,
    )
    df.to_csv(file_name, index=False)
