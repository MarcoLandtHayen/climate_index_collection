import weakref
from numpy.testing import assert_almost_equal
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray
import pandas as pd
import math
from pathlib import Path

from climate_index_collection.reductions import (
    grouped_mean_weighted,
    monthly_mean_weighted,
    monthly_mean_unweighted,
    monthly_anomalies_unweighted,
    monthly_anomalies_weighted,
)
from climate_index_collection.data_loading import load_data_set, VARNAME_MAPPING

# ========
# CREATE TEST DATA PARAMETERS AND FUNCTIONS
# ========

lon = np.array([120,140,150])
lat = np.array([-10, -5, 0])

def create_data_array(values,  dim, dim_name) :
    """
    This function creates test DataArrays from given lat, lon, group and groupname and weights.
    -----
    Parameters:
        lat: numpy.adarray, list
        lon: numpy.adarray, list
        values: numpy.adarray
        group: numpy.adarray, list
        groupname: str
    """
    # create dummy dataset 
    data = DataArray(values, 
                     dims=(dim_name, 
                           "lat", 
                           "lon"), 
                     coords={dim_name : dim, 
                             "lat": lat, 
                             'lon': lon})
    return data

def create_weight_array(wei, dim, dim_name) :    
    weights = DataArray(wei, 
                        dims=(dim_name), 
                        coords={dim_name : dim})
    return weights 

def create_mean_array(mean, group_unique, group_name) :    
    weights = DataArray(mean, 
                 dims=(group_name, "lat", "lon"), 
                 coords={group_name : group_unique, "lat": lat, 'lon': lon})
    return weights 

# ----------
# First test DataArray
# ----------
weights_1 = [1, 2, 3, 2]
dim_1 = ['a','b','a','c']
dim_name_1 = group_name_1 = "group" 
group_unique_1 = np.unique(dim_1)

np.random.seed(100)
values_1 = np.random.randint(0,2, (len(dim_1), len(lat), len(lon)) ).astype(float)
values_1[0,0,0] = np.nan
data_1 = create_data_array(values = values_1, 
                            dim = dim_1,
                            dim_name = dim_name_1)
weights_1 = create_weight_array(wei = weights_1,
                                dim = dim_1,
                                dim_name = dim_name_1)


# Should be the correct values
weighted_mean_1 = np.array(
      [[[0.  , 0.75, 0.25],
        [1.  , 0.25, 0.25],
        [0.  , 0.75, 0.75]],

       [[0.  , 0.  , 1.  ],
        [0.  , 0.  , 0.  ],
        [0.  , 1.  , 0.  ]],

       [[1.  , 0.  , 0.  ],
        [1.  , 0.  , 0.  ],
        [1.  , 1.  , 1.  ]]])

weighted_mean_1 = create_mean_array(mean = weighted_mean_1,
                                        group_unique = group_unique_1,
                                        group_name = group_name_1)

# ----------
# Second test DataArray
# ----------
dim_2 = pd.to_datetime(["2020-02-13", "2021-06-13", "2021-08-13", "2022-02-13"])
dim_name_2 = "time"
weights_2 = dim_2.days_in_month
weights_2 = create_weight_array(wei = weights_2, 
                                  dim = dim_2, 
                                  dim_name=dim_name_2)
group_unique_2 = np.unique(dim_2.month)
group_name_2 = "month"
values_2 = np.array(
      [[[np.nan,  0., 57.],
        [57., 57., 57.],
        [ 0.,  0.,  57.]],

       [[ 0.,  0., 57.],
        [ 0.,  0.,  0.],
        [ 0., 57.,  0.]],

       [[ 0., 57.,  0.],
        [57.,  0.,  0.],
        [ 0., 57., 57.]],

       [[57.,  0.,  0.],
        [57.,  0.,  0.],
        [57., 57., 57.]]])


data_2 = create_data_array(values = values_2, 
                           dim = dim_2, 
                           dim_name=dim_name_2)
# also create a dataset
dataset = data_2.to_dataset(dim=None, name="test", promote_attrs=False)

weighted_mean_2 = np.array(
      [[[57.,  0., 29.],
        [57., 29., 29.],
        [28., 28., 57.]],

       [[ 0.,  0., 57.],
        [ 0.,  0.,  0.],
        [ 0., 57.,  0.]],

       [[ 0., 57.,  0.],
        [57.,  0.,  0.],
        [ 0., 57., 57.]]])
weighted_mean_2 = create_mean_array(mean = weighted_mean_2,
                                        group_unique=group_unique_2,
                                        group_name = group_name_2)
unweighted_mean_2 = np.array(
      [[[57,  0., 28.5,],
        [57, 28.5, 28.5,],
        [28.5, 28.5,  57]],

       [[ 0.,  0., 57.],
        [ 0.,  0.,  0.],
        [ 0., 57.,  0.]],

       [[ 0., 57.,  0.],
        [57.,  0.,  0.],
        [ 0., 57., 57.]]])
unweighted_mean_2 = create_mean_array(mean = unweighted_mean_2,
                                        group_unique=group_unique_2,
                                        group_name = group_name_2)


# monthly anomalies for the data. 
# nans will be included!
weighted_anomalies_2 = np.    array(
        [[[np.nan,   0.,  28.],
            [  0.,  28.,  28.],
            [-28., -28.,   0.]],

           [[  0.,   0.,   0.],
            [  0.,   0.,   0.],
            [  0.,   0.,   0.]],

           [[  0.,   0.,   0.],
            [  0.,   0.,   0.],
            [  0.,   0.,   0.]],

           [[  0.,   0., -29.],
            [  0., -29., -29.],
            [ 29.,  29.,   0.]]])
weighted_anomalies_2 = create_data_array(values = weighted_anomalies_2, 
                                  dim = dim_2, 
                                  dim_name=dim_name_2)

unweighted_anomalies_2 = np.array(
     [[[np.nan,   0. ,  28.5],
        [  0. ,  28.5,  28.5],
        [-28.5, -28.5,   0. ]],

       [[  0. ,   0. ,   0. ],
        [  0. ,   0. ,   0. ],
        [  0. ,   0. ,   0. ]],

       [[  0. ,   0. ,   0. ],
        [  0. ,   0. ,   0. ],
        [  0. ,   0. ,   0. ]],

       [[  0. ,   0. , -28.5],
        [  0. , -28.5, -28.5],
        [ 28.5,  28.5,   0. ]]])
unweighted_anomalies_2 = create_data_array(values = unweighted_anomalies_2, 
                                  dim = dim_2, 
                                  dim_name=dim_name_2)

# ========
# Basic tests with the test data created above
# ========

@pytest.mark.parametrize("data, weights, dim     , groupby_dim, should",[ 
    (data_1, weights_1, group_name_1 , group_name_1 , weighted_mean_1),
    (data_2, weights_2, "time" , "time.month" , weighted_mean_2),
                         ])
def test_grouped_mean_weighted(data, weights, dim, groupby_dim, should):
    """Checks if the groupby weighting function gives proper results."""
    result = grouped_mean_weighted(dobj=data, weights= weights, dim = dim, groupby_dim= groupby_dim)
    assert result.equals(should)

@pytest.mark.parametrize("data_array, should",[(data_2, weighted_mean_2)])
def test_monthly_mean_weighted(data_array, should):
    """Checks if the monthly mean weighted function gives proper results."""
    result = monthly_mean_weighted(dobj=data_array)
    assert result.equals(should)

@pytest.mark.parametrize("data_array, should",[(data_2, unweighted_mean_2)])
def test_monthly_mean_unweighted(data_array, should):
    """Checks if the monthly mean unweighted function gives proper results."""
    result = monthly_mean_unweighted(dobj=data_array)
    assert result.equals(should)

# Tests for anomaly functions
# check if the results are correct
@pytest.mark.parametrize("data_array, should",[(data_2, weighted_anomalies_2)])
def test_monthly_anomalies_weighted(data_array, should):
    """Checks if the monthly anomalies weighted function gives proper results."""
    result = monthly_anomalies_weighted(dobj=data_array)
    assert result.equals(should)

@pytest.mark.parametrize("data_array, should",[(data_2, unweighted_anomalies_2)])
def test_monthly_anomalies_unweighted(data_array, should):
    """Checks if the monthly anomalies unweighted function gives proper results."""
    result = monthly_anomalies_unweighted(dobj=data_array)
    assert result.equals(should)

# ========
# Test if the means created by the anomalies functions are close to zero.
# Note: The weighted function will only be zero if one creates the monthly mean values
# ========

def variable_close_to_desired(original, actual, desired = 0, relative_tolerance = 1e-5):
    """
    Checks if all spatial point from the input array "actual" are close to the desired value.
    This assert is based on the numpy.testing.assert_almost_equal.
    The decimal used for this is obtained based on the relative tolerance given.
    It is calculated as described below
        1. absolute accuracy = (maximum - minimum) * relative tolerance.
        2. decimal = -1 * Order(absolute accuracy)
    Maximum and minimum are derived from the "original" DataArray 
    which was used for the calculation of the DataArray "actual".
    For some purpose it might make sense to hand original and actual the same DataArray.
    
    Parameters
    ----------
    original: xarray.DataArray
        DataArray containing field from which e.g. the anomalies were calulated.
    actual: xarray.DataArray
        Dataset with same variables as orignial but containing the values, 
        which shall be close to the desired value.
        For instance the monthly anomalies derived from original
    desired: float
        Desired value.
        Default to 0.
    relative_tolerance : float
        Relative tolerance which shall be used to derive the decimal accuracy.
        Default to 1e-5.
    
    """
    min_value = original.min().values
    max_value = original.max().values
    absolute_tolerance = (max_value-min_value)*relative_tolerance
    # calculate the desired decimal accuracy as 
    # -1 * Order(absolute accuracy)
    decimal = -1 * math.floor(math.log(absolute_tolerance, 10))
    assert_almost_equal(
            actual = actual.values,
            desired = desired,
            decimal=decimal)

def all_variables_close_to_desired(original, actual, desired = 0, relative_tolerance = 1e-5):
    """
    Checks if all variable from the input DataSet "actual" are close to the desired value.
    This assert is based on the numpy.testing.assert_almost_equal.
    The decimal used for this is obtained based on the relative tolerance given.
    It is calculated as described below
        1. absolute accuracy = (maximum - minimum) * relative tolerance.
        2. decimal = -1 * Order(absolute accuracy)
    Maximum and minimum are derived from the "original" DataSet for each variable independently. 
    The "original" should be the DataSet from which "actual" was calculated.
    For some purpose it might make sense to hand original and actual the same DataArray.

    
    Parameters
    ----------
    original: xarray.DataSet
        Dataset containing field from which e.g. the anomalies were calulated.
    actual: xarray.DataSet
        Dataset with same variables as orignial but containing the values, 
        which shall be close to the desired value.
        For instance the monthly anomalies derived from original
    desired: float
        Desired value.
        Default to 0.
    relative_tolerance : float
        Relative tolerance which shall be used to derive the decimal accuracy 
        for each variable individually.
        Default to 1e-5.
    
    """
    for variable in original.keys() :
        if "time" in variable:
            continue
        variable_close_to_desired(original[variable], actual[variable], desired = 0, relative_tolerance = 1e-5)

@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys())+[dataset])
def test_monthly_anomalies_weighted_zeromean(source_name, relative_tolerance = 1e-5):
    """Checks if the mean of the monthly anomalies weighted are all close to 0 using
    all_variables_close_to_desired which is based on numpy.testing.assert_almost_equal.
    The test will be performed for each spatial gridpoint individually.
    
    NOTE: 
        As this is a weighted mean, the values of the anomalies will not sum up to zero with a convenient .mean("time").
        One should better check if the monthly_mean_weighted of the anomalies sums up to zero for each month.
    
    Parameters
    ----------
    source_name: str
        Test dataset name.
    relative_tolerance : float
        Relative tolerance which shall be used to derive the decimal accuracy 
        for each variable individually.
        Default to 1e-5.

    """
    if isinstance(source_name, xr.Dataset) :
        data_set = source_name
    else : 
        # Load test data
        TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
        data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
    
    anomalies = monthly_anomalies_weighted(dobj=data_set)
    anomalies_mean = monthly_mean_weighted(anomalies)
    
    # check if the mean of the anomalies is close to 0 with desired decimal accuracy 
    # derived based of relative tolerance for each variable indepenently 
    all_variables_close_to_desired(
        original = data_set, 
        actual= anomalies_mean, 
        desired = 0, 
        relative_tolerance = relative_tolerance)

@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys())+[dataset])
def test_monthly_anomalies_unweighted_zeromean(source_name, relative_tolerance = 1e-5):
    """Checks if the mean of the monthly anomalies unweighted are all close to 0 using
    all_variables_close_to_desired which is based on numpy.testing.assert_almost_equal
    The test will be performed for each spatial gridpoint individually.
    
    Parameters
    ----------
    source_name: str or DataSet
        Test dataset name.
        Or DataSet directly
    relative_tolerance : float
        Relative tolerance which shall be used to derive the decimal accuracy 
        for each variable individually.
        Default to 1e-5.

    """

    if isinstance(source_name, xr.Dataset) :
        data_set = source_name
    else : 
        # Load test data
        TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
        data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)
    
    anomalies = monthly_anomalies_unweighted(dobj=data_set)
    anomalies_mean = anomalies.mean("time")
    # check if the mean of the anomalies is close to 0 with desired decimal accuracy 
    # derived based of relative tolerance for each variable indepenently 
    all_variables_close_to_desired(
        original = data_set, 
        actual= anomalies_mean, 
        desired = 0, 
        relative_tolerance = relative_tolerance)
