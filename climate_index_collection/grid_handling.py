import numpy as np
import xarray as xr


def get_lat_bins(lat, north_lat=90.0, south_lat=-90.0):
    """Get latitude bins for non-uniform latitude.

    NOTE: Either needs to be applied before cropping the global data
    or you need to set the north_lat and south_lat keywords accordingly.

    1. Construct location of faces between grid boxes using the
       average of nearest neighbors.
    2. Add +/- 90.0 degrees (or other North / South lat).
    3. Calc box size using diff.

    Parameters
    ----------
    lat: xr.DataArray
        Latitude array.
    north_lat: float
        Northernmost lat face. Defaults to 90.0
    south_lat: float
        Southernmost lat face. Defaults to -90.0

    Returns
    -------
    xr.DataArray
        Latitude extent.

    """
    lat = lat.sortby(lat, ascending=True)
    lat_faces = np.array(
        [
            -90.0,
        ]
        + list((lat.data[1:] + lat.data[:-1]) / 2.0)
        + [
            90.0,
        ]
    )
    lat_bins = 0 * lat + (lat_faces[1:] - lat_faces[:-1])
    return lat_bins


def get_xy_weights(lat=None, lon=None, north_lat=90.0, south_lat=-90.0):
    """Calculate xy weights for uniform lon and non-uniform lat.

    NOTE: Either needs to be applied before cropping the global data
    or you need to set the north_lat and south_lat keywords accordingly.

    Parameters
    ----------
    lat: xr.DataArray
        Latitude array.
    lon: xr.DataArray
        Longitude array.
    north_lat: float
        Northernmost lat face. Defaults to 90.0
    south_lat: float
        Southernmost lat face. Defaults to -90.0

    Returns
    -------
    xr.DataArray
        Weights. (Not normalized).

    """
    dlat = get_lat_bins(lat, north_lat=north_lat, south_lat=south_lat)
    dy = dlat
    dlon = abs(lon.diff("lon").isel(lon=0))
    dx = dlon * np.cos(np.deg2rad(lat))
    return dx * dy


def scale_field_xy_weights(dobj, north_lat=90.0, south_lat=-90.0):
    """Scale field according to lat / lon weights.

    NOTE: Either needs to be applied before cropping the global data
    or you need to set the north_lat and south_lat keywords accordingly.

    Parameters
    ----------
    dobj: xr.DataArray or xr.Dataset
        Has dimensional coords 'lat' and 'lon'.
    north_lat: float
        Northernmost lat face. Defaults to 90.0
    south_lat: float
        Southernmost lat face. Defaults to -90.0

    Returns
    -------
    xr.DataArray or xr.Dataset
        Scaled with xy weights.

    """
    weights = get_xy_weights(
        lat=dobj.coords["lat"],
        lon=dobj.coords["lon"],
        north_lat=north_lat,
        south_lat=south_lat,
    )
    # return weights * dobj
    return dobj
