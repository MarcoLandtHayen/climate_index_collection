import weakref
from climate_index_collection.reductions import (
    mean_unweighted,
    mean_weighted,
    stddev_weighted,
    stddev_unweighted,
    variance_weighted,
    variance_unweighted,
)

import pytest

import xarray as xr
import numpy as np


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
        [[1, 2, 3, 4], [4, 3, np.nan, 1]], dims=("t", "x"), name="data",
    )
    return data


@pytest.fixture
def example_weights_01():
    weights = xr.DataArray([1, 2,], dims=("t"), name="weights",)
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
