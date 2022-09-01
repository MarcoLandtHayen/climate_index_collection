import numpy as np
import xarray as xr

from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import split, unary_union


# -----------------------------
# Basic statistical moments
# -----------------------------


def mean_unweighted(dobj, dim=None):
    """Calculate an unweighted mean for one or more dimensions.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    dim: str or list
        Dimension name or list of dimension names.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Averaged data. Has the same variable name(s) as dobj.

    """
    return dobj.mean(dim)


def mean_weighted(dobj, weights=None, dim=None):
    """Calculate a weighted mean.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    weights: xarray.DataArray
        Contains the weights.
    dim: str or list
        Dimension name or list of dimension names.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Averaged data. Has the same variable name(s) as dobj.

    """
    numerator = (dobj * weights).sum(dim)
    denominator = weights.where(~dobj.isnull()).sum(dim)
    weighted_mean = numerator / denominator
    return weighted_mean


def variance_weighted(dobj, weights=None, dim=None):
    """Calculate a weighted variance.

    This uses a weighted mean <.>_w and defines variance as
    var[a] = <a**2>_w - (<a>_w)**2.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    weights: xarray.DataArray
        Contains the weights.
    dim: str or list
        Dimension name or list of dimension names.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Variance data. Has the same variable name(s) as dobj.

    """
    mean_of_squares = mean_weighted(dobj**2, weights=weights, dim=dim)
    square_of_means = mean_weighted(dobj, weights=weights, dim=dim) ** 2
    variance = mean_of_squares - square_of_means
    return variance


def variance_unweighted(dobj, dim=None):
    """Calculate an unweighted variance.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    dim: str or list
        Dimension name or list of dimension names.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Variance data. Has the same variable name(s) as dobj.

    """
    return dobj.var(dim)


def stddev_weighted(dobj, weights=None, dim=None):
    """Calculate a weighted stddev.

    This uses a weighted mean <.>_w, defines variance as
    var[a] = <a**2>_w - (<a>_w)**2, and std dev as var***0.5.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    weights: xarray.DataArray
        Contains the weights.
    dim: str or list
        Dimension name or list of dimension names.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Stddev data. Has the same variable name(s) as dobj.

    """
    variance = variance_weighted(dobj, weights=weights, dim=dim)
    std_dev = variance**0.5
    return std_dev


def stddev_unweighted(dobj, dim=None):
    """Calculate an unweighted stddev.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    dim: str or list
        Dimension name or list of dimension names.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Stddev data. Has the same variable name(s) as dobj.

    """
    return dobj.std(dim)


def grouped_mean_weighted(dobj, weights=None, dim=None, groupby_dim=None):
    """Calculate a weighted mean depending on a xarray.groupby call
    and the corresponding groupby dimension.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    weights: xarray.DataArray
        Contains the weights.
    dim: str or list
        Dimension name or list of dimension names.
    groupby_dim: str or list
        Dimension name ofr list of the dimension names used to groupby.
        e.g. time.month

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Averaged data. Has the same variable name(s) as dobj.
        Will include the the groupby_dim as new dimension.
        Will not include the dimension dim over which the mean is calculated.
    """
    # numerator and denominator both need to be grouped !!
    numerator = (dobj * weights).groupby(groupby_dim).sum(dim)
    denominator = weights.where(~dobj.isnull()).groupby(groupby_dim).sum(dim)
    weighted_mean = numerator / denominator
    return weighted_mean


# -----------------------------
# Monthly climatology analysis
# -----------------------------


def monthly_weights(dobj, normalize=True):
    """Calculate the monthly weights including leap years.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    normalize: bool+
        Normalization of the weights performed if set to True.
        No normalization is set to False.
        Default is False.
    Returns
    -------
    xarray.DataArray
        Weight for each timestep depending on the month.
        If normalized is True: normalized length of the month,
        else : length of the month as int.

    """
    # The following setting will allow to calculate the monthly mean weighted
    dim = "time"
    groupby_dim = "time.month"
    month_length = dobj.time.dt.days_in_month
    num_groups = len(np.unique(dobj.time.dt.month))

    # Check if weights shall be normalized
    if normalize:
        weights = month_length.groupby(groupby_dim) / month_length.groupby(
            groupby_dim
        ).sum("time")
        # Test that the sum of the weights for each season is 1.0
        np.testing.assert_allclose(
            weights.groupby(groupby_dim).sum().values, np.ones(num_groups)
        )
    else:
        weights = month_length

    return weights


def monthly_mean_unweighted(dobj):
    """Calculates the monthly mean values of a dataset.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Monthly mean data. Has the same variable name(s) as dobj.
        Dimension 'time' will be removed.
        Dimension 'month' is gained. Int values, starting with 1 for January.
    """
    return mean_unweighted(dobj=dobj.groupby("time.month"), dim="time")


def monthly_anomalies_unweighted(dobj):
    """Calculates the monthly anomalies from the monthly climatology of a dataset.
    The monthly climatology is calculated using "monthly_mean"

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Monthly anomalies from the monthly climatology of the original data.
        Has the same variable name(s) as dobj.
    """
    # Note: the dobj needs to be grouped by the same group_dim as the DataArray returned by monthly_mean_unweighted!
    return (dobj.groupby("time.month") - monthly_mean_unweighted(dobj)).drop_vars(
        "month"
    )


def spatial_mask(
    dobj,
    lat_south=None,
    lat_north=None,
    lon_west=None,
    lon_east=None,
    lon_name="lon",
    lat_name="lat",
):
    """Mask over lat and lon range.

    Parameters
    ----------
    dobj: xarray.DataArray
        Data for which the mask will be created.
    lat_south: float
        Southern latitude bound. If "None", only the other lat bound will be applied.
    lat_north: float
        Northern latitude bound. If "None", only the other lat bound will be applied.
    lon_west: float
        Western longitude bound. If "None", only the other lon bound will be applied.
    lon_east: float
        Eastern longitude bound. If "None", only the other lon bound will be applied.
    lat_name: str
        Name of the latitude coordinate. Defaults to "lat".
    lon_name: str
        Name of the longitude coordinate. Defaults to "lon".

    Returns
    -------
    dobj:
        Mask for given lat and lon range.

    """
    # extract coords
    lat = dobj.coords[lat_name]
    lon = dobj.coords[lon_name]

    # maybe standardize lon
    lon = lon % 360
    if lon_west is not None:
        lon_west = lon_west % 360
    if lon_east is not None:
        lon_east = lon_east % 360

    # catch unset lat and lon bounds
    if lon_west is None:
        lon_west = -np.inf
    if lon_east is None:
        lon_east = np.inf
    if lat_south is None:
        lat_south = -np.inf
    if lat_north is None:
        lat_north = np.inf

    # coords
    # check if lon range does not cross the greenwhich meridian at 0°W.
    if lon_west <= lon_east:
        return (
            (lat >= lat_south)
            & (lat <= lat_north)
            & (lon >= lon_west)
            & (lon <= lon_east)
        )
    # otherwise us 'or' instead of 'and' for the lon logical operation.
    else:
        return (
            (lat >= lat_south)
            & (lat <= lat_north)
            & ((lon >= lon_west) | (lon <= lon_east))
        )


def area_mean_weighted(
    dobj,
    lat_south=None,
    lat_north=None,
    lon_west=None,
    lon_east=None,
    lon_name="lon",
    lat_name="lat",
    lat_extent_name=None,
    lon_extent_name=None,
):
    """Area average over lat and lon range.

    Parameters
    ----------
    dobj: xarray.DataArray
        Data for which the area average will be computed.
    lat_south: float
        Southern latitude bound. If "None", only the other lat bound will be applied.
    lat_north: float
        Northern latitude bound. If "None", only the other lat bound will be applied.
    lon_west: float
        Western longitude bound. If "None", only the other lon bound will be applied.
    lon_east: float
        Eastern longitude bound. If "None", only the other lon bound will be applied.
    lat_name: str
        Name of the latitude coordinate. Defaults to "lat".
    lon_name: str
        Name of the longitude coordinate. Defaults to "lon".
    lat_extent_name: str
        Name of the lat extent. Defaults to None.
    lon_extent_name: str
        Name of the lon extent. Defaults to None.

    Returns
    -------
    dobj:
        Area averaged input data.

    """
    # extract coords
    lat = dobj.coords[lat_name]
    lon = dobj.coords[lon_name]

    # coords
    mask = spatial_mask(
        dobj=dobj,
        lat_south=lat_south,
        lat_north=lat_north,
        lon_west=lon_west,
        lon_east=lon_east,
        lon_name=lon_name,
        lat_name=lat_name,
    )

    # do we have dimensional coords?
    have_dimensional_coords = (lat.dims[0] == lat_name) & (lon.dims[0] == lon_name)

    # prepare spatial coords for final sum
    if have_dimensional_coords:
        spatial_dims = [lat_name, lon_name]
    else:
        spatial_dims = list(set(lat.dims) + set(lon.dims))

    # prepare weights
    if have_dimensional_coords:
        # will only handle equidistant lons and lats here...
        dlon = abs(lon.diff(lon_name).mean())
        dlat = abs(lat.diff(lat_name).mean())
        dx = dlon * np.cos(np.deg2rad(lat))
        dy = dlat
        weights = dx * dy
    else:
        lat_extent = dobj.coords[lat_extent_name]
        lon_extent = dobj.coords[lon_extent_name]
        weights = lat_extent * lon_extent

    # weighted averaging
    averaged = (dobj * weights).where(mask).sum(spatial_dims) / weights.where(mask).sum(
        spatial_dims
    )

    return averaged


def eof_weights(dobj):
    """Empirical orthogonal functions (EOFs) can be thought of as the eigen-vectors of the spatial covariance matrix.
    Grid cells can be thought of as representing spatial averages which are low-pass filtering the raw signals and dampen the variance by a factor proportional to the square root of the cell size.
    For equidistant latitude/longitude grids the area weights are proportional to cos(latitude).
    Before applying Singular Value Decomposition (SVD), input data needs to be multiplied with the square root of the weights.


    Parameters
    ----------
    dobj: xarray.DataArray
        Contains the original input data.

    Returns
    -------
    xarray.DataArray
        Square root of weights needed to pre-process input data for SVD.
    """
    return np.sqrt(np.cos(np.deg2rad(dobj.coords["lat"])))


def polygon_prime_meridian(pg):
    """
    Transforms shapely Polygons or MultiPolygons defined in [180W, 180E) coords into [0E,360E) coords.
    Takes care of Polygons crossing the prime meridan.
    Polygon points are expected be (lon, lat) tuples.

    Parameters
    ----------
    pg: shaply Polygon or shapely MultiPolygon
        Polygon including the area wanted.

    Returns
    -------
    shapely MultPolygon
        shaply MultiPolygon containing at least one Polygon.
    """

    # handle empty Polygons and MultiPolygons
    if pg.is_empty:
        return MultiPolygon([pg])

    # create prime_meridian to eventually split the polygon
    prime_meridian = LineString([(0, 90), (0, -90)])
    pg_split = split(pg, prime_meridian)

    # create a list containing all Polygons given by the split operation
    # polygons on the negative (western) side of the prime meridian are translated into new coords, by adding 360 to the lon values.
    pg_list = []
    for temp_pg in pg_split.geoms:
        # check if the poygon minx is negative and if, add 360 to it.
        if temp_pg.bounds[0] < 0:
            temp_pg = translate(temp_pg, xoff=360)
        pg_list += [temp_pg]

    # create the multipolygon existing in [0E, 360E) coords from the list of polygons
    result = unary_union(pg_list)

    # for consistency always return MultiPolygon
    if type(result) is not MultiPolygon:
        # convert Polygon to MultiPolygon
        return MultiPolygon([result])
    else:
        return result


def polygon2mask(dobj, pg, lat_name="lat", lon_name="lon"):
    """
    This funciton creates a mask for a given DataArray or DataSet based on a shapely Polygon or MultiPolygon.
    Polygon points are expected be (lon, lat) tuples.
    The dobj is expected to have lon values in [0E, 360E) coords.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    pg: shaply Polygon or shapely MultiPolygon
        Polygon including the area wanted.
    lat_name: str
        Name of the latitude coordinate. Defaults to "lat".
    lon_name: str
        Name of the longitude coordinate. Defaults to "lon".

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mask for given Polygon on dobj grid.
    """

    # handle prime meridian crossing Polygons and transform [180W, 180E) coords into [0E,360E) coords
    pg = polygon_prime_meridian(pg)

    # create the mask
    lon_2d, lat_2d = xr.broadcast(dobj.coords[lon_name], dobj.coords[lat_name])

    mask = xr.DataArray(
        np.reshape(
            [
                pg.contains(Point(_lon, _lat)) | pg.boundary.contains(Point(_lon, _lat))
                for _lon, _lat in zip(np.ravel(lon_2d), np.ravel(lat_2d))
            ],
            lon_2d.shape,
        ),
        dims=lon_2d.dims,
        coords=lon_2d.coords,
    )
    # transpose to have lon on x and lat on y axes
    mask = mask.transpose()
    return mask
