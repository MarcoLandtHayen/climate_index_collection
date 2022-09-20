import weakref

import numpy as np
import pytest
import xarray as xr

from numpy.testing import assert_almost_equal
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import split, unary_union

from climate_index_collection.reductions import (
    area_mean_weighted_polygon_selection,
    mean_unweighted,
    mean_weighted,
    polygon2mask,
    polygon2mask_only_primemeridian,
    polygon_prime_meridian,
    polygon_split_arbitrary,
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


@pytest.mark.parametrize(
    "polygon_split_function", [polygon_prime_meridian, polygon_split_arbitrary]
)
def test_polygon_split_functions_no_crossing_polygon(polygon_split_function):
    """
    Check case where input is a
    - Polygon which
    - does not cross prime meridian.
    The result needs to be
    - correct and
    - be a MultiPolygon
    """

    pg = Polygon([(10, 50), (5, 50), (5, -50), (10, -50)])
    result = polygon_split_function(pg)
    should_result = Polygon([(10, 50), (5, 50), (5, -50), (10, -50)])
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


@pytest.mark.parametrize(
    "polygon_split_function", [polygon_prime_meridian, polygon_split_arbitrary]
)
def test_polygon_split_functions_no_crossing_multipolygon(polygon_split_function):
    """
    Check case where input is a
    - MultiPolygon which
    - does not cross the prime meridian.
    The result needs to be
    - correct and
    - be a MultiPolygon
    """

    pg = MultiPolygon(
        [
            Polygon([(10, 50), (5, 50), (5, 10), (10, 10)]),
            Polygon([(10, -10), (5, -10), (5, -50), (5, -50)]),
        ]
    )
    result = polygon_split_function(pg)
    should_result = MultiPolygon(
        [
            Polygon([(10, 50), (5, 50), (5, 10), (10, 10)]),
            Polygon([(10, -10), (5, -10), (5, -50), (5, -50)]),
        ]
    )
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


@pytest.mark.parametrize(
    "polygon_split_function", [polygon_prime_meridian, polygon_split_arbitrary]
)
def test_polygon_split_functions_crossing_polygon(polygon_split_function):
    """
    Check case where input is a
    - Polygon which
    - does cross the prime meridian.
    The result needs to be
    - correct and
    - be a MultiPolygon
    """

    pg = Polygon([(10, 50), (-10, 50), (-10, -50), (10, -50)])
    result = polygon_split_function(pg)
    should_result = unary_union(
        [
            Polygon([(360, 50), (350, 50), (350, -50), (360, -50)]),
            Polygon([(10, 50), (0, 50), (0, -50), (10, -50)]),
        ]
    )
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


@pytest.mark.parametrize(
    "polygon_split_function", [polygon_prime_meridian, polygon_split_arbitrary]
)
def test_polygon_split_functions_crossing_multipolygon(polygon_split_function):
    """
    Check case where input is a
    - MultiPolygon which
    - does cross the prime meridian.
    The result needs to be
    - correct and
    - be a MultiPolygon
    """

    pg = MultiPolygon(
        [
            Polygon([(10, 50), (-10, 50), (-10, -50), (10, -50)]),
            Polygon([(50, -10), (180, -10), (180, -50), (50, -50)]),
        ]
    )
    result = polygon_split_function(pg)
    should_result = MultiPolygon(
        [
            Polygon([(10, 50), (0, 50), (0, -50), (10, -50)]),
            Polygon([(50, -10), (180, -10), (180, -50), (50, -50)]),
            Polygon([(360, 50), (350, 50), (350, -50), (360, -50)]),
        ]
    )
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


@pytest.mark.parametrize(
    "polygon_split_function", [polygon_prime_meridian, polygon_split_arbitrary]
)
def test_polygon_split_functions_crossing_multipolygon_overlap(polygon_split_function):
    """
    Check case where input is a
    - MultiPolygon which
    - does cross the prime meridian
    - and is defined in coords [180W, 360E).
    The last point needs to function to handle overlapping Polygons after the spilt operation.
    The result needs to be
    - correct and
    - handles the resulting overlap of the Polygons created by the split and
    - be a MultiPolygon
    """

    pg = MultiPolygon(
        [
            Polygon([(10, 50), (-10, 50), (-10, -50), (10, -50)]),
            Polygon([(50, -10), (355, -10), (355, -50), (50, -50)]),
        ]
    )
    result = polygon_split_function(pg)
    should_result = MultiPolygon(
        [
            Polygon([(10, 50), (0, 50), (0, -50), (10, -50)]),
            Polygon(
                [
                    (360, 50),
                    (350, 50),
                    (350, -10),
                    (50, -10),
                    (50, -50),
                    (360, -50),
                ]
            ),
        ]
    )
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


@pytest.mark.parametrize(
    "polygon_split_function", [polygon_prime_meridian, polygon_split_arbitrary]
)
def test_polygon_split_functions_whole_earth(polygon_split_function):
    """
    Check case where input is a Polygon which cointains the whole earth.
    The result needs to be
    - correct and
    - has same area as before and
    - be a MultiPolygon
    """
    pg = Polygon([(-180, 90), (180, 90), (180, -90), (-180, -90)])
    should_result = Polygon([(0, 90), (360, 90), (360, -90), (0, -90)])
    result = polygon_split_function(pg)
    assert pg.area == result.area
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


@pytest.mark.parametrize(
    "polygon_split_function", [polygon_prime_meridian, polygon_split_arbitrary]
)
def test_polygon_split_functions_empty(polygon_split_function):
    """
    Check case where input is an empty Polygon or Multipolygon.
    The result needs to be
    - empty
    - be a MultiPolygon
    """
    pg = Polygon()
    mpg = MultiPolygon()
    pg_result = polygon_split_function(pg)
    mpg_result = polygon_prime_meridian(mpg)
    assert all(result.is_empty for result in [pg_result, mpg_result])
    assert all(type(result) is MultiPolygon for result in [pg_result, mpg_result])


def test_polygon_split_arbitrary_ulr():
    """
    Check case a rectangle Polygon needs to be split along upper, left and right boundary.
    """
    pg = Polygon(
        [
            (-10, 100),
            (10, 100),
            (10, 80),
            (-10, 80),
        ]
    )
    should_result = MultiPolygon(
        [
            Polygon([(0, 90), (10, 90), (10, 80), (0, 80)]),
            Polygon([(0, -90), (10, -90), (10, -80), (0, -80)]),
            Polygon([(350, 90), (360, 90), (360, 80), (350, 80)]),
            Polygon([(350, -90), (360, -90), (360, -80), (350, -80)]),
        ]
    )
    result = polygon_split_arbitrary(pg)
    assert pg.area == result.area
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


def test_polygon_split_arbitrary_default():
    """
    Check case a rectangle Polygon needs to be split along all boundaries.

    In the following explanation the input MultiPolygon (iMP) and result MultiPolygon (rMP) is sketched.
    x : inside one of the Polygons.
    0 : not inside one of the Polygons.
    | : lon bounds
    - : lat bounds
    Input MultiPolygon looks like this
    x x o o o o
     ---------
    x|x o o o|o
    o|o o o o|o
    o|o o o o|o
    o|o o o x|x
     ---------
    o o o o x x

    the result will then be
    o o o o o o
     ---------
    o|x o o x|o
    o|o o o o|o
    o|o o o o|o
    o|x o o x|o
     ---------
    o o o o o o
    """
    pg = MultiPolygon(
        [
            Polygon([(-10, 100), (10, 100), (10, 80), (-10, 80)]),
            Polygon([(370, -100), (350, -100), (350, -80), (370, -80)]),
        ]
    )

    should_result = MultiPolygon(
        [
            Polygon([(0, 90), (10, 90), (10, 80), (0, 80)]),
            Polygon([(0, -90), (10, -90), (10, -80), (0, -80)]),
            Polygon([(350, 90), (360, 90), (360, 80), (350, 80)]),
            Polygon([(350, -90), (360, -90), (360, -80), (350, -80)]),
        ]
    )
    result = polygon_split_arbitrary(pg)
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


def test_polygon_split_arbitrary_notdefault():
    """
    Same as in test_polygon_split_arbitrary_default but bounds and polygon input values divided by 10.
    """
    pg = MultiPolygon(
        [
            Polygon([(-1, 10), (1, 10), (1, 8), (-1, 8)]),
            Polygon([(37, -10), (35, -10), (35, -8), (37, -8)]),
        ]
    )

    should_result = MultiPolygon(
        [
            Polygon([(0, 9), (1, 9), (1, 8), (0, 8)]),
            Polygon([(0, -9), (1, -9), (1, -8), (0, -8)]),
            Polygon([(35, 9), (36, 9), (36, 8), (35, 8)]),
            Polygon([(35, -9), (36, -9), (36, -8), (35, -8)]),
        ]
    )
    result = polygon_split_arbitrary(pg, lon_min=0, lon_max=36, lat_min=-9, lat_max=9)
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


def test_polygon_split_arbitrary_notdefault2():
    """
    Same test_polygon_split_arbitrary_notdefault but with bounds
    lon_min=-1, lon_max=0, lat_min=9, lat_max=10
    """
    pg = MultiPolygon(
        [
            Polygon([(-1, 10), (1, 10), (1, 8), (-1, 8)]),
            Polygon([(37, -10), (35, -10), (35, -8), (37, -8)]),
        ]
    )

    should_result = Polygon([(0, 10), (-1, 10), (-1, 9), (0, 9)])
    result = polygon_split_arbitrary(pg, lon_min=-1, lon_max=0, lat_min=9, lat_max=10)
    assert result.equals(should_result)
    assert type(result) is MultiPolygon


@pytest.mark.parametrize(
    "polygon_mask_function", [polygon2mask, polygon2mask_only_primemeridian]
)
def test_polygon2mask_across_dateline(polygon_mask_function):
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
    pg = Polygon([(30, 90), (270, 90), (270, -90), (30, -90)])
    mask = polygon_mask_function(data_set, pg)
    assert mask.astype(int).sum().data[()] == 2
    assert all(m == mt for m, mt in zip([False, True, True], mask.squeeze().data))


@pytest.mark.parametrize(
    "polygon_mask_function", [polygon2mask, polygon2mask_only_primemeridian]
)
def test_polygon2mask_across_zero_meridian(polygon_mask_function):
    """Check case where lon W/E bounds are ordered in interval [-180, 180)."""
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
    pg = Polygon([(30, 90), (-180, 90), (-180, -90), (30, -90)])
    mask = polygon_mask_function(data_set, pg)
    assert mask.astype(int).sum().data[()] == 2
    assert all(m == mt for m, mt in zip([True, False, True], mask.squeeze().data))


@pytest.mark.parametrize(
    "polygon_mask_function", [polygon2mask, polygon2mask_only_primemeridian]
)
def test_polygon2mask_point_on_boundary(polygon_mask_function):
    """Check case where a point lies on the boundary of the polygon.
    Here it is the Point (0, 0) is situated on the boundary."""
    data_set = xr.Dataset(
        coords={
            "lat": [
                0.0,
                -10.0,
            ],
            "lon": [
                0.0,
                120.0,
                240.0,
            ],
        }
    )
    pg = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

    mask = polygon_mask_function(data_set, pg)

    mask_should = np.array([[True, False, False], [False, False, False]])

    assert mask.astype(int).sum().data[()] == 1
    assert all(m == mt for m, mt in zip(mask_should.flatten(), mask.data.flatten()))


@pytest.mark.parametrize(
    "polygon_mask_function", [polygon2mask, polygon2mask_only_primemeridian]
)
def test_polygon2mask_multipolygon(polygon_mask_function):
    """Check case where a point lies on the boundary of the polygon.
    Here it is the Point (0, 0) is situated on the boundary."""
    data_set = xr.Dataset(
        coords={
            "lat": [
                0.0,
                -10.0,
            ],
            "lon": [
                0.0,
                120.0,
                240.0,
            ],
        }
    )
    pg1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    pg2 = Polygon([(110, -10), (130, -10), (130, -15), (110, -15)])
    pg = unary_union([pg1, pg2])
    mask = polygon_mask_function(data_set, pg)
    mask_should = np.array([[True, False, False], [False, True, False]])
    assert mask.astype(int).sum().data[()] == 2
    assert all(m == mt for m, mt in zip(mask_should.flatten(), mask.data.flatten()))


def test_area_mean_weighted_polygon_selection():
    """Ensure that polygon selection results in correct example average."""
    #
    # data:
    #    4 5 6
    #    1 2 3
    #
    # polygon covers:
    #    . . .
    #    x x .
    #
    # resulting average = (1 + 2) / 2
    #
    test_data = xr.DataArray(
        [[1, 2, 3], [4, 5, 6]],
        name="data",
        dims=("lat", "lon"),
        coords={"lat": [5.0, 6.0], "lon": [10.0, 20.0, 30.0]},
    )
    test_polygon_lon_lat = Polygon([(9.0, 4.5), (21.0, 4.5), (21.0, 5.5), (9.0, 5.5)])

    assert_almost_equal(
        actual=area_mean_weighted_polygon_selection(
            test_data, polygon_lon_lat=test_polygon_lon_lat
        ).data[()],
        desired=(1 + 2) / 2,
        decimal=3,
    )


def test_area_mean_weighted_polygon_no_selection():
    """Ensure that polygon selection results in correct example average."""
    #
    # data:
    #    4 5 6
    #    1 2 3
    #
    # polygon covers everything:
    #    x x x
    #    x x x
    #
    # resulting average = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5
    #
    test_data = xr.DataArray(
        [[1, 2, 3], [4, 5, 6]],
        name="data",
        dims=("lat", "lon"),
        coords={"lat": [5.0, 6.0], "lon": [10.0, 20.0, 30.0]},
    )
    test_polygon_lon_lat = None

    assert_almost_equal(
        actual=area_mean_weighted_polygon_selection(
            test_data, polygon_lon_lat=test_polygon_lon_lat
        ).data[()],
        desired=3.5,
        decimal=3,
    )
