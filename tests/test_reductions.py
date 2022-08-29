import weakref

import numpy as np
import pytest
import xarray as xr

from climate_index_collection.reductions import (
    mean_unweighted,
    mean_weighted,
    spatial_mask,
    stddev_unweighted,
    stddev_weighted,
    variance_unweighted,
    variance_weighted,
)


# Test data
#
# data = [[1, 2, 3, 4], [4, 3, NAN, 1]]
# weights = [[1, 1, 1, 1], [2, 2, 2, 2]]
#
# weighted_mean = [3.0, 8/3, 3.0, 2.0]
# unweighted_mean = [2.5, 2.5, 3.0, 2.5]
# weighted_var = [2.0, 2/9, 0.0, 2.0]
# unweighted_var = [2.25, 0.25, 0.0, 2.25]
#


@pytest.fixture
def example_dataset_01():
    data = xr.DataArray(
        [[1, 2, 3, 4], [4, 3, np.nan, 1]],
        dims=("t", "x"),
        name="data",
    )
    return data


@pytest.fixture
def example_weights_01():
    weights = xr.DataArray(
        [
            1,
            2,
        ],
        dims=("t"),
        name="weights",
    )
    return weights


def test_weighted_mean(example_dataset_01, example_weights_01):
    reduced = mean_weighted(example_dataset_01, weights=example_weights_01, dim="t")
    np.testing.assert_allclose(np.array([3.0, 8 / 3, 3.0, 2.0]), reduced.data)


def test_unweighted_mean(example_dataset_01):
    reduced = mean_unweighted(example_dataset_01, dim="t")
    np.testing.assert_allclose(np.array([2.5, 2.5, 3, 2.5]), reduced.data)


def test_weighted_var(example_dataset_01, example_weights_01):
    reduced = variance_weighted(example_dataset_01, weights=example_weights_01, dim="t")
    np.testing.assert_allclose(np.array([2.0, 2 / 9, 0.0, 2.0]), reduced.data)


def test_weighted_std(example_dataset_01, example_weights_01):
    reduced = stddev_weighted(example_dataset_01, weights=example_weights_01, dim="t")
    np.testing.assert_allclose(np.array([2.0, 2 / 9, 0.0, 2.0]) ** 0.5, reduced.data)


def test_unweighted_var(example_dataset_01):
    reduced = variance_unweighted(example_dataset_01, dim="t")
    np.testing.assert_allclose(np.array([2.25, 0.25, 0.0, 2.25]), reduced.data)


def test_unweighted_std(example_dataset_01):
    reduced = stddev_unweighted(example_dataset_01, dim="t")
    np.testing.assert_allclose(np.array([2.25, 0.25, 0.0, 2.25]) ** 0.5, reduced.data)


def test_spatial_mask_across_dateline():
    """Check case where lon W/E bounds are ordered in interval [0,360)."""
    data_set = xr.Dataset(
        coords={
            "lat": [
                0.0,
            ],
            "lon": [
                0.0,
                120.0,
                240.0,
            ],
        }
    )
    mask = spatial_mask(
        dobj=data_set,
        lat_south=-90.0,
        lat_north=90.0,
        lon_west=30,
        lon_east=270,
    )
    assert mask.astype(int).sum().data[()] == 2
    assert all(m == mt for m, mt in zip([False, True, True], mask.squeeze().data))


def test_spatial_mask_across_zero_meridian():
    """Check case wher lon W/E bounds are ordered in interval [-180, 180)."""
    data_set = xr.Dataset(
        coords={
            "lat": [
                0.0,
            ],
            "lon": [
                0.0,
                120.0,
                240.0,
            ],
        }
    )
    mask = spatial_mask(
        dobj=data_set,
        lat_south=-90.0,
        lat_north=90.0,
        lon_west=270,
        lon_east=30.0,
    )
    assert mask.astype(int).sum().data[()] == 1
    assert all(m == mt for m, mt in zip([True, False, False], mask.squeeze().data))


def test_spatial_mask_no_lon_masking():
    """Check that not setting lon bounds works as intended."""
    data_set = xr.Dataset(
        coords={
            "lat": [
                0.0,
            ],
            "lon": [
                0.0,
                120.0,
                240.0,
            ],
        }
    )
    mask = spatial_mask(
        dobj=data_set,
        lat_south=-90.0,
        lat_north=90.0,
        lon_west=None,
        lon_east=None,
    )
    assert mask.astype(int).sum().data[()] == 3
    assert all(m == mt for m, mt in zip([True, True, True], mask.squeeze().data))


def test_spatial_mask_partial_lon_masking():
    """Check that not setting all lon bounds works as intended."""
    data_set = xr.Dataset(
        coords={
            "lat": [
                0.0,
            ],
            "lon": [
                0.0,
                120.0,
                240.0,
            ],
        }
    )
    mask = spatial_mask(
        dobj=data_set,
        lat_south=-90.0,
        lat_north=90.0,
        lon_west=None,
        lon_east=180.0,
    )
    assert mask.astype(int).sum().data[()] == 2
    assert all(m == mt for m, mt in zip([True, True, False], mask.squeeze().data))


def test_spatial_mask_no_lat_masking():
    """Check that not setting lat bounds works as intended."""
    data_set = xr.Dataset(
        coords={
            "lat": [-60.0, 0.0, 60.0],
            "lon": [
                30.0,
            ],
        }
    )
    mask = spatial_mask(
        dobj=data_set,
        lat_south=None,
        lat_north=None,
        lon_west=0.0,
        lon_east=60.0,
    )
    assert mask.astype(int).sum().data[()] == 3
    assert all(m == mt for m, mt in zip([True, True, True], mask.squeeze().data))


def test_spatial_mask_partial_lat_masking():
    """Check that not setting all lat bounds works as intended."""
    data_set = xr.Dataset(
        coords={
            "lat": [-60.0, 0.0, 60.0],
            "lon": [
                30.0,
            ],
        }
    )
    mask = spatial_mask(
        dobj=data_set,
        lat_south=-30.0,
        lat_north=None,
        lon_west=0.0,
        lon_east=60.0,
    )
    assert mask.astype(int).sum().data[()] == 2
    assert all(m == mt for m, mt in zip([False, True, True], mask.squeeze().data))


def test_spatial_mask_no_bounds():
    """Check that not setting any bounds works as intended."""
    data_set = xr.Dataset(
        coords={
            "lat": [-60.0, 0.0, 60.0],
            "lon": [
                0.0,
                120.0,
                240.0,
            ],
        }
    )
    mask = spatial_mask(
        dobj=data_set,
        lat_south=None,
        lat_north=None,
        lon_west=None,
        lon_east=None,
    )
    assert mask.astype(int).sum().data[()] == 9
    assert all(list(mask.data.flatten()))
