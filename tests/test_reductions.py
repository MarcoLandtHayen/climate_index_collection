import weakref
from climate_index_collection.reductions import (
    mean_unweighted,
    mean_weighted,
    stddev_weighted,
    variance_weighted,
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


@pytest.fixture
def example_dataset():
    data = xr.DataArray(
        [[1, 2, 3, 4], [4, 3, np.nan, 1]], dims=("t", "x"), name="data",
    )
    return data


@pytest.fixture
def example_weights():
    weights = xr.DataArray([1, 2,], dims=("t"), name="weights",)
    return weights


def test_weighted_mean(example_dataset, example_weights):
    reduced = mean_weighted(example_dataset, weights=example_weights, dim="t")
    np.testing.assert_allclose(np.array([3.0, 8 / 3, 3.0, 2.0]), reduced.data)


def test_unweighted_mean(example_dataset):
    reduced = mean_unweighted(example_dataset, dim="t")
    np.testing.assert_allclose(np.array([2.5, 2.5, 3, 2.5]), reduced.data)


def test_weighted_var(example_dataset, example_weights):
    reduced = variance_weighted(example_dataset, weights=example_weights, dim="t")
    np.testing.assert_allclose(np.array([2.0, 2 / 9, 0.0, 2.0]), reduced.data)


def test_weighted_std(example_dataset, example_weights):
    reduced = stddev_weighted(example_dataset, weights=example_weights, dim="t")
    np.testing.assert_allclose(np.array([2.0, 2 / 9, 0.0, 2.0]) ** 0.5, reduced.data)
