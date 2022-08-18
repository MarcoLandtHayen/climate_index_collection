from enum import Enum
from functools import partial

import numpy as np
import scipy as sp
import xarray as xr

from .reductions import (
    area_mean_weighted,
    mean_unweighted,
    monthly_anomalies_unweighted,
)


def southern_annular_mode(data_set, slp_name="sea-level-pressure"):
    """Calculate the southern annular mode (SAM) index.

    This follows [Gong and Wang, 1999] <https://doi.org/10.1029/1999GL900003> in defining
    the southern annular mode index using zonally averaged sea-level pressure at 65°S and
    40°S.

    It differs from the definition of [Gong and Wang, 1999] in that it uses the raw time
    series of zonally averaged sea-level pressure and then only normalizes (zero mean,
    unit standard deviation) of the difference of zonally avearged sea-level pressure at
    65°S and 40°S.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SLP field.
    slp_name: str
        Name of the Sea-Level Pressure field. Defaults to "sea-level-pressure".

    Returns
    -------
    xarray.DataArray
        Time series containing the SAM index.

    """
    slp = data_set[slp_name]

    slp40S = mean_unweighted(slp.sel(lat=-40, method="nearest"), dim="lon")
    slp65S = mean_unweighted(slp.sel(lat=-65, method="nearest"), dim="lon")

    slp_diff = slp40S - slp65S

    SAM_index = (slp_diff - slp_diff.mean("time")) / slp_diff.std("time")
    SAM_index = SAM_index.rename("SAM")

    return SAM_index


def southern_annular_mode_pc(data_set, geopoth_name="geopotential-height"):
    """Calculate the principal component based southern annular mode (SAM) index.

    Following https://climatedataguide.ucar.edu/climate-data/marshall-southern-annular-mode-sam-index-station-based
    this index is obtained as Principle Component (PC) time series of the leading Empirical Orthogonal Function (EOF)
    of monthly geopotential height anomalies over parts of the Southern hemisphere.

    Note: There is no unique definition for pc-based SAM index. Here we use geopotential height for 500 hPa and take
    Southern hemisphere from 20°S to 90°S.

    Computation is done as follows:
    1. Compute geopotential height anomalies over parts of the Southern hemisphere.
    2. Flatten spatial dimensions and subtract mean in time.
    3. Perform Singular Value Decomposition.
    4. Normalize Principal Components.
    5. Obtain SAM index as PC time series related to leading EOF.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a geopotential height field.
    geopoth_name: str
        Name of the geopotential height field. Defaults to "geopotential-height".

    Returns
    -------
    xarray.DataArray
        Time series containing the SAM index.

    """

    mask = data_set.coords["lat"] <= -20

    geopoth = data_set[geopoth_name].where(mask)
    climatology = geopoth.groupby("time.month").mean("time")
    geopoth = (geopoth.groupby("time.month") - climatology).drop("month")

    geopoth_flat = geopoth.stack(tmp_space=("lat", "lon")).dropna(dim="tmp_space")

    pc, s, eof = sp.linalg.svd(
        geopoth_flat - geopoth_flat.mean(axis=0), full_matrices=False
    )

    pc_std = pc.std(axis=0)
    pc /= pc_std

    SAM_index = xr.DataArray(
        pc[:, 0], dims=("time"), coords={"time": geopoth_flat["time"]}
    )

    SAM_index = SAM_index.rename("SAM_PC")

    return SAM_index


def north_atlantic_oscillation(data_set, slp_name="sea-level-pressure"):
    """Calculate the station based North Atlantic Oscillation (NAO) index

    This uses station-based sea-level pressure closest to Reykjavik (64°9'N, 21°56'W) and
    Ponta Delgada (37°45'N, 25°40'W) and, largely following [Hurrel, 1995]
    <https://doi.org/10.1126/science.269.5224.676>, defines the north-atlantic oscillation index
    as the difference of normalized in Reykjavik and Ponta Delgada without normalizing the
    resulting timeseries again. (This means that the north atlantic oscillation presented here
    has vanishing mean because both minuend and subtrahend have zero mean, but no unit
    standard deviation.)

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SLP field.
    slp_name: str
        Name of the Sea-Level Pressure field. Defaults to "sea-level-pressure".

    Returns
    -------
    xarray.DataArray
        Time series containing the NAO index.

    """
    slp = data_set[slp_name]

    slp_northern_station = slp.sel(lat=64, lon=338, method="nearest")
    slp_southern_station = slp.sel(lat=38, lon=334, method="nearest")

    slp_northern_station_norm = (
        slp_northern_station - slp_northern_station.mean("time")
    ) / slp_northern_station.std("time")
    slp_southern_station_norm = (
        slp_southern_station - slp_southern_station.mean("time")
    ) / slp_southern_station.std("time")

    NAO_index = slp_northern_station_norm - slp_southern_station_norm
    NAO_index = NAO_index.rename("NAO")

    return NAO_index


def north_atlantic_oscillation_pc(data_set, slp_name="sea-level-pressure"):
    """Calculate the principal component based North Atlantic Oscillation (NAO) index

    Following
    https://climatedataguide.ucar.edu/climate-data/hurrell-north-atlantic-oscillation-nao-index-pc-based
    this index is obtained as Principle Component (PC) time series of the leading Empirical Orthogonal Function (EOF)
    of monthly sea-level pressure anomalies over the Atlantic sector, 20°-80°N, 90°W-40°E.

    Computation is done as follows:
    1. Compute sea level pressure anomalies over Atlantic sector.
    2. Flatten spatial dimensions and subtract mean in time.
    3. Perform Singular Value Decomposition.
    4. Normalize Principal Components.
    5. Obtain NAO index as PC time series related to leading EOF.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SLP field.
    slp_name: str
        Name of the Sea-Level Pressure field. Defaults to "sea-level-pressure".

    Returns
    -------
    xarray.DataArray
        Time series containing the NAO index.

    """

    mask = (
        (data_set.coords["lat"] >= 20)
        & (data_set.coords["lat"] <= 80)
        & ((data_set.coords["lon"] >= 270) | (data_set.coords["lon"] <= 40))
    )

    slp = data_set[slp_name].where(mask)
    climatology = slp.groupby("time.month").mean("time")
    slp = (slp.groupby("time.month") - climatology).drop("month")

    slp_flat = slp.stack(tmp_space=("lat", "lon")).dropna(dim="tmp_space")

    pc, s, eof = sp.linalg.svd(slp_flat - slp_flat.mean(axis=0), full_matrices=False)

    pc_std = pc.std(axis=0)
    pc /= pc_std

    NAO_index = xr.DataArray(pc[:, 0], dims=("time"), coords={"time": slp_flat["time"]})

    NAO_index = NAO_index.rename("NAO_PC")

    return NAO_index


def el_nino_southern_oscillation_34(data_set, sst_name="sea-surface-temperature"):
    """Calculate the El Nino Southern Oscillation 3.4 index (ENSO 3.4)

    Following https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
    the index is derived from equatorial pacific sea-surface temperature anomalies in a box
    borderd by 5°S, 5°N, 120°W and 170°W.

    Computation is done as follows:
    1. Compute area averaged total SST from Niño 3.4 region.
    2. Compute monthly climatology for area averaged total SST from Niño 3.4 region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Note: Usually the index is smoothed by taking some rolling mean over 5 months before
    normalizing. We omit the rolling mean here and directly take sst anomaly index instead,
    to preserve the information in full detail. And as climatology we use the complete time span,
    since we deal with model data.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.
    sst_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the ENSO 3.4 index.

    """
    sst_nino34 = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=-5,
        lat_north=5,
        lon_west=190,
        lon_east=240,
    )

    climatology = sst_nino34.groupby("time.month").mean("time")

    std_dev = sst_nino34.std("time")

    ENSO34_index = (sst_nino34.groupby("time.month") - climatology) / std_dev
    ENSO34_index = ENSO34_index.rename("ENSO34")

    return ENSO34_index


def north_atlantic_sea_surface_salinity(data_set, sss_name="sea-surface-salinity"):
    """Calculate North-Atlantic Sea-Surface Salinity index.

    Following https://doi.org/10.1126/sciadv.1501588
    the index is derived from Atlantic sea-surface salinity (SSS) anomalies in a box
    spanning 50W-15W, 25N-50N.

    Note that in the original ref, there was a smaller box spanning 50W-40W, 37.5N-50N
    cut away from the one used here. We skip this here, for now.

    Computation is done as follows:
    1. Calculate the weighted area average of SSS over 50W-15W, 25N-50N.
    2. Standardize time series.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SSS field.
    sss_name: str
        Name of the Sea-Surface Salinity field. Defaults to "sea-surface-salinity".

    Returns
    -------
    xarray.DataArray
        Time series containing the NASSS index.

    """
    sss = data_set[sss_name]

    sss_box_ave = area_mean_weighted(
        sss,
        lat_south=25,
        lat_north=50,
        lon_west=-50,
        lon_east=-15,
    )

    NASSS = (sss_box_ave - sss_box_ave.mean("time")) / sss_box_ave.std("time")

    NASSS = NASSS.rename("NASSS")

    return NASSS


def sea_air_surface_temperature_anomaly_north(data_set, sat_name="sea-air-temperature"):
    """Sea Air Surface Temperature Anomaly (SASTA) index ,
    for the northern hemisphere.

    This follows https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies
    Land and Sea temperature data is used.
    The Anomalies are climatoligical anomalies (monthly) relative to the whole time period of the data_set.

    Computation is done as follows:
    1. Compute area averaged total SAT for the hemisphere.
    2. Compute monthly climatology for area averaged SAT for the hemisphere.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SAT field.
    slp_name: str
        Name of the Sea-Air Temperature field. Defaults to "sea-air-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SASTA index for northern hemisphere.

    """
    sea_air_temperature = area_mean_weighted(
        dobj=data_set[sat_name], lat_south=0, lat_north=90, lon_west=0, lon_east=360
    )

    sea_air_temp_anom = monthly_anomalies_unweighted(sea_air_temperature)
    SASTAI_North = sea_air_temp_anom.rename("SASTAI-north")

    return SASTAI_North


def sea_air_surface_temperature_anomaly_south(data_set, sat_name="sea-air-temperature"):
    """Sea Air Surface Temperature Anomaly (SASTA) index ,
    for the southern hemisphere.

    This follows https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies
    Land and Sea temperature data is used.
    The Anomalies are climatoligical anomalies (monthly) relative to the whole time period of the data_set.

    Computation is done as follows:
    1. Compute area averaged total SAT for the hemisphere.
    2. Compute monthly climatology for area averaged SAT for the hemisphere.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SAT field.
    slp_name: str
        Name of the Sea-Air Temperature field. Defaults to "sea-air-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SASTA index for northern hemisphere.

    """
    sea_air_temperature = area_mean_weighted(
        dobj=data_set[sat_name], lat_south=-90, lat_north=0, lon_west=0, lon_east=360
    )

    sea_air_temp_anom = monthly_anomalies_unweighted(sea_air_temperature)
    SASTAI_South = sea_air_temp_anom.rename("SASTAI-south")

    return SASTAI_South


class ClimateIndexFunctions(Enum):
    """Enumeration of all index functions.

    We can use this to, e.g., iterate over all defined indices.

    >>> for index_func in ClimateIndexFunctions:
    ...     print(index_func.name)
    southern_annular_mode
    north_atlantic_oscillation
    ...

    """

    southern_annular_mode = partial(southern_annular_mode)
    southern_annular_mode_pc = partial(southern_annular_mode_pc)
    north_atlantic_oscillation = partial(north_atlantic_oscillation)
    north_atlantic_oscillation_pc = partial(north_atlantic_oscillation_pc)
    el_nino_southern_oscillation_34 = partial(el_nino_southern_oscillation_34)
    north_atlantic_sea_surface_salinity = partial(north_atlantic_sea_surface_salinity)
    sea_air_surface_temperature_anomaly_north = partial(
        sea_air_surface_temperature_anomaly_north
    )
    sea_air_surface_temperature_anomaly_south = partial(
        sea_air_surface_temperature_anomaly_south
    )

    @classmethod
    def get_all_names(cls):
        """Return tuple with all known index names."""
        return tuple(cls.__members__.keys())
