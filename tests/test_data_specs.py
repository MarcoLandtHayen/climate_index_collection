from pathlib import Path
import numpy as np
from xarray import DataArray
import itertools
import pytest

from climate_index_collection.data_loading import VARNAME_MAPPING, load_data_set
from climate_index_collection.data_specs import get_spacial_dimension_specs, latitude_longitude_specs, sel_latitude_longitude_slice


# ========
# CREATE TEST DATA
# ========
test_data_0 = np.array([0,1,2,4,6,8,11,12,13])
test_result_0 = dict(
        mini = 0,
        mini_pos = np.array([0]),
        maxi = 13,
        maxi_pos = np.array([8]),
        size = 9, 
        diff_constant = False,
        diff_sign = True,
        diff_change_position = np.array([0,2,5,6]),
        diff_change_values = np.array([1,2,3,1]),
        diff = np.array([1,1,2,2,2,3,1,1]),
        )
test_data_1 = np.array([4.5, 3, 1.5, 0, -1.5])
test_result_1 = dict(
        mini = -1.5,
        mini_pos = np.array([4]),
        maxi = 4.5,
        maxi_pos = np.array([0]),
        size = 5, 
        diff_constant = True,
        diff_sign = False,
        diff_change_position = np.array([]),
        diff_change_values = np.array([]),
        diff = np.array([-1.5, -1.5, -1.5, -1.5]),
        )
test_data_2 = np.array([-3, -9, 2, -9, np.nan])
test_result_2 = dict(
        mini = -9,
        mini_pos = np.array([1,3]),
        maxi = 2,
        maxi_pos = np.array([2]),
        size = 5, 
        diff_constant = False,
        diff_sign = None,
        diff_change_position = np.array([0,1,2,3]),
        diff_change_values = np.array([-6, 11, -11, np.nan]),
        diff = np.array([-6, 11, -11, np.nan]),
        )

# -------
# Create two datasets containing random reproducable int values and set latitude and longitude
# -------
# DATASET 0
# set latitude and longitude and corresponding should values
lat_0 = test_data_0
lat_specs_should_0 = test_result_0
lon_0 = test_data_1
lon_specs_should_0 = test_result_1
# get random int from seed
np.random.seed(100)
val_0 = np.random.randint(0,9, (len(lat_0), len(lon_0)) )
# create dummy dataset 
data_0 = DataArray(val_0, dims=("lat", "lon"), coords={"lat": lat_0, 'lon': lon_0})
# create boundaries and slices for the corresponding dataset
lat_bounds_0 = (2,5)
lon_bounds_0 = (-10,2)
lat_slice_should_0 = slice(2,5)
lon_slice_should_0 = slice(2, -10)

# DATASET 1
# set latitude and longitude and corresponding should values
lat_1 = test_data_1
lat_specs_should_1 = test_result_1
lon_1 = test_data_0
lon_specs_should_1 = test_result_0
# get random int from seed
np.random.seed(100)
val_1 = np.random.randint(0, 9, (len(lat_1), len(lon_1)) )
# create dummy dataset 
data_1 = DataArray(val_1, dims=("lat", "lon"), coords={"lat": lat_1, 'lon': lon_1})
# create boundaries and slices for the corresponding dataset
lat_bounds_1 = (2,5)
lon_bounds_1 = (-10,2)
lat_slice_should_1 = slice(5,2)
lon_slice_should_1 = slice(-10, 2)

# ------------
# Comparison dict function
# ------------

def compare_dict_items(d1, d2):
    """
    This function checks if two dictonaries contain the same keys and values in first layer depth.
    It does not work recursively, thus only checks first layer depth!
    For numpy.ndarrays it can handle numpy.nan values.
    
    -------
    Parameters
    d1 : dict
        first dictonary
    d2 : dict
        second dictonary
    -------
    Return
    bool : 
        True if d1 and d2 do contain same keys and values in first layer depth
        False if d1 and d2 do NOT contain same keys and values in first layer depth
    """
    sorted_keys = lambda d : sorted(d.keys())
    # check if all keys are the same
    if not sorted_keys(d1) ==  sorted_keys(d2) :
        return False
    
    comparison = []
    for key in d1.keys() :
        d1_item = d1[key]
        d2_item = d2[key]
        try :
            comparison.append(
                np.array_equal(
                    d1_item, 
                    d2_item, 
                    equal_nan = True) 
            )
        except TypeError as te:
            comparison.append(d1_item == d2_item)
    return not False in comparison

# =========
# TEST ROUTINES
# =========

@pytest.mark.parametrize("d1, d2, should",[ 
        (test_result_0, test_result_1, False),
        (test_result_0, test_result_0, True),
        (test_result_0, test_result_2, False),
        (test_result_2, test_result_2, True),
                         ])
def test_compare_dict_items(d1, d2, should):
    """Checks if compare_dict_items gives proper results."""
    print(compare_dict_items(d1 = d1, d2 = d2))
    assert should == compare_dict_items(d1 = d1, d2 = d2)

@pytest.mark.parametrize("data, result_should", [
        (test_data_0, test_result_0),
        (test_data_1, test_result_1),
        (test_data_2, test_result_2),
        ])
def test_get_spacial_dimension_specs(data, result_should):
    """Check if the dictonary contains all keya and if they are correct."""
    result = get_spacial_dimension_specs(data=data)    
    # check if all items are the same 
    # (also np.ndarray including np.nan)
    assert compare_dict_items(result, result_should)

@pytest.mark.parametrize("data, lat_specs_should, lon_specs_should", [
    (data_0, lat_specs_should_0, lon_specs_should_0),
    (data_1, lat_specs_should_1, lon_specs_should_1)
    ])
def test_latitude_longitude_specs(data, lat_specs_should, lon_specs_should):
    """Check if the dictonary contains all keys and if the values are correct."""
    result = latitude_longitude_specs(dobj = data)
    lat_specs = result['lat_specs']
    lon_specs = result['lon_specs']
    # compare should and is resulting dictonaries 
    # Latitude
    assert compare_dict_items(lat_specs, lat_specs_should)
    # Longitude
    assert compare_dict_items(lon_specs, lon_specs_should)
    

@pytest.mark.parametrize("data, lat_bounds, lon_bounds, lat_slice_should, lon_slice_should", [
                        (data_0, lat_bounds_0, lon_bounds_0, lat_slice_should_0, lon_slice_should_0),
                        (data_1, lat_bounds_1, lon_bounds_1, lat_slice_should_1, lon_slice_should_1),
                        ])
def test_sel_latitude_longitude_slice(data, lat_bounds, lon_bounds, lat_slice_should, lon_slice_should):
    """Check if the dictonary contains all keys and if the values are correct."""
    LatSlice, LonSlice = sel_latitude_longitude_slice(dobj = data, LatBounds=lat_bounds, LonBounds=lon_bounds)
    # check if resulting slices are correct 
    assert all((
        LatSlice == lat_slice_should,
        LonSlice == lon_slice_should
        )
        )
