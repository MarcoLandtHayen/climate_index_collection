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
    mean = mean_weighted(example_dataset, weights=example_weights, dim="t")
    np.testing.assert_allclose(np.array([3.0, 8 / 3, 3.0, 2.0]), mean.data)


def test_weighted_var(example_dataset, example_weights):
    mean = mean_weighted(example_dataset, weights=example_weights, dim="t")
    np.testing.assert_allclose(np.array([3.0, 8 / 3, 3.0, 2.0]), mean.data)
