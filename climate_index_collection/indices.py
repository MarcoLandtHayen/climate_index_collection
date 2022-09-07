from enum import Enum
from functools import partial

import numpy as np
import scipy as sp
import xarray as xr

from shapely.geometry import Polygon

from .reductions import (
    area_mean_weighted,
    area_mean_weighted_polygon_selection,
    eof_weights,
    mean_unweighted,
    monthly_anomalies_unweighted,
    polygon_prime_meridian,
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
    Southern hemisphere from 20°S to 90°S. And to have the EOFs truly orthogonal, we need to take the area of the grid cells
    into account. For equidistant latitude/longitude grids the area weights are proportional to cos(latitude).
    Before applying Singular Value Decomposition, input data is multiplied with the square root of the weights.

    Computation is done as follows:
    1. Compute geopotential height anomalies over parts of the Southern hemisphere.
    2. Flatten spatial dimensions and subtract mean in time.
    3. Perform Singular Value Decomposition.
    4. Normalize Principal Components.
    5. Obtain SAM index as PC time series related to leading EOF.
    6. Restore EOF patterns.
    7. Use positive pole of leading EOF to invert index values - if necessary.

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

    geopoth = data_set[geopoth_name]
    geopoth = geopoth * eof_weights(geopoth)
    geopoth = geopoth.where(mask)
    climatology = geopoth.groupby("time.month").mean("time")
    geopoth = (geopoth.groupby("time.month") - climatology).drop("month")

    geopoth_flat = geopoth.stack(tmp_space=("lat", "lon")).dropna(dim="tmp_space")

    pc, s, eof = sp.linalg.svd(
        geopoth_flat - geopoth_flat.mean(axis=0), full_matrices=False
    )

    pc_std = pc.std(axis=0)
    pc /= pc_std

    SAM_index = xr.DataArray(pc[:, 0], dims=("time"), coords={"time": geopoth["time"]})

    eofs = geopoth.stack(tmp_space=("lat", "lon")).copy()[0:1, :]
    eofs[0:1, eofs[0].notnull().values] = (
        eof * pc_std[:, np.newaxis] * s[:, np.newaxis]
    )[0:1, :]
    eofs = eofs.unstack(dim="tmp_space").rename(**{"time": "mode"})

    mask_pos = (eofs[0].coords["lat"] <= -20) & (eofs[0].coords["lat"] >= -40)
    if eofs[0].where(mask_pos).mean(("lat", "lon")) < 0:
        SAM_index.values = -SAM_index.values

    SAM_index = SAM_index.rename("SAM_PC")

    return SAM_index


def southern_oscillation(data_set, slp_name="sea-level-pressure"):
    """Calculate the station based Southern Oscillation Index (SOI)

    This uses sea-level pressure (SLP) closest to Tahiti (17°41'S, 149°27'W) and
    Darwin (12°27'S, 130°50'O) and, following https://doi.org/10.1126/science.269.5224.676,
    defines SOI index from difference of normalized SLP anomalies in Tahiti and Darwin.
    The resulting time series is then itself normalized over the whole time span.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SLP field.
    slp_name: str
        Name of the Sea-Level Pressure field. Defaults to "sea-level-pressure".

    Returns
    -------
    xarray.DataArray
        Time series containing the SOI index.

    """
    slp = data_set[slp_name]

    slp_tahiti = slp.sel(lat=-18, lon=211, method="nearest")
    slp_darwin = slp.sel(lat=-12, lon=131, method="nearest")

    slp_tahiti_norm = (slp_tahiti - slp_tahiti.mean("time")) / slp_tahiti.std("time")
    slp_darwin_norm = (slp_darwin - slp_darwin.mean("time")) / slp_darwin.std("time")

    slp_diff = slp_tahiti_norm - slp_darwin_norm

    SOI_index = slp_diff / slp_diff.std("time")
    SOI_index = SOI_index.rename("SOI")

    return SOI_index


def north_atlantic_oscillation(data_set, slp_name="sea-level-pressure"):
    """Calculate the station based North Atlantic Oscillation (NAO) index

    This uses station-based sea-level pressure closest to Reykjavik (64°9'N, 21°56'W) and
    Ponta Delgada (37°45'N, 25°40'W) and, largely following
    https://doi.org/10.1126/science.269.5224.676, defines the north-atlantic oscillation index
    as the difference of normalized in Reykjavik and Ponta Delgada without normalizing the
    resulting time series again. (This means that the north atlantic oscillation presented here
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

    Note: To have the EOFs truly orthogonal, we need to take the area of the grid cells into account.
    For equidistant latitude/longitude grids the area weights are proportional to cos(latitude).
    Before applying Singular Value Decomposition, input data is multiplied with the square root of the weights.

    Computation is done as follows:
    1. Compute sea level pressure anomalies over Atlantic sector.
    2. Flatten spatial dimensions and subtract mean in time.
    3. Perform Singular Value Decomposition.
    4. Normalize Principal Components.
    5. Obtain NAO index as PC time series related to leading EOF.
    6. Restore leading EOF pattern.
    7. Use positive pole of leading EOF to correct index sign - if necessary.


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

    slp = data_set[slp_name]
    slp = slp * eof_weights(slp)
    slp = slp.where(mask)
    climatology = slp.groupby("time.month").mean("time")
    slp = (slp.groupby("time.month") - climatology).drop("month")

    slp_flat = slp.stack(tmp_space=("lat", "lon")).dropna(dim="tmp_space")

    pc, s, eof = sp.linalg.svd(slp_flat - slp_flat.mean(axis=0), full_matrices=False)

    pc_std = pc.std(axis=0)
    pc /= pc_std

    NAO_index = xr.DataArray(pc[:, 0], dims=("time"), coords={"time": slp["time"]})

    eofs = slp.stack(tmp_space=("lat", "lon")).copy()[0:1, :]
    eofs[0:1, eofs[0].notnull().values] = (
        eof * pc_std[:, np.newaxis] * s[:, np.newaxis]
    )[0:1, :]
    eofs = eofs.unstack(dim="tmp_space").rename(**{"time": "mode"})

    mask_pos = eofs[0].coords["lat"] >= 60
    if eofs[0].where(mask_pos).mean(("lat", "lon")) < 0:
        NAO_index.values = -NAO_index.values

    NAO_index = NAO_index.rename("NAO_PC")

    return NAO_index


def el_nino_southern_oscillation_12(data_set, sst_name="sea-surface-temperature"):
    """Calculate the El Nino Southern Oscillation 1+2 index (ENSO 1+2)

    Following https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
    the index is derived from equatorial pacific sea-surface temperature (SST) anomalies in a box
    bordered by 0°S - 10°S and 90°W - 80°W. This translates to -10°N - 0°N and 270°E - 280°E.
    The Niño 1+2 region is the smallest and eastern-most of the Niño regions,
    and corresponds with the region of coastal South America where El Niño was first recognized by the local populations.
    This index tends to have the largest variance of the Niño SST indices.

    Computation is done as follows:
    1. Compute area averaged total SST from Niño 1+2 region.
    2. Compute monthly climatology for area averaged total SST from Niño 1+2 region.
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
        Time series containing the ENSO 1+2 index.

    """
    sst_nino12 = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=-10,
        lat_north=0,
        lon_west=270,
        lon_east=280,
    )

    climatology = sst_nino12.groupby("time.month").mean("time")

    std_dev = sst_nino12.std("time")

    ENSO12_index = (sst_nino12.groupby("time.month") - climatology) / std_dev
    ENSO12_index = ENSO12_index.rename("ENSO12")

    return ENSO12_index


def el_nino_southern_oscillation_3(data_set, sst_name="sea-surface-temperature"):
    """Calculate the El Nino Southern Oscillation 3 index (ENSO 3)

    Following https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
    the index is derived from equatorial pacific sea-surface temperature (SST) anomalies in a box
    bordered by 5°S - 5°N and 150°W - 90°W. This translates to -5°N - 5°N and 210°E - 270°E.
    This region was once the primary focus for monitoring and predicting El Niño, but researchers later
    learned that the key region for coupled ocean-atmosphere interactions for ENSO lies further west.
    Hence, the Niño 3.4 became favored for defining El Niño and La Niña events.

    Computation is done as follows:
    1. Compute area averaged total SST from Niño 3 region.
    2. Compute monthly climatology for area averaged total SST from Niño 3 region.
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
        Time series containing the ENSO 3 index.

    """
    sst_nino3 = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=-5,
        lat_north=5,
        lon_west=210,
        lon_east=270,
    )

    climatology = sst_nino3.groupby("time.month").mean("time")

    std_dev = sst_nino3.std("time")

    ENSO3_index = (sst_nino3.groupby("time.month") - climatology) / std_dev
    ENSO3_index = ENSO3_index.rename("ENSO3")

    return ENSO3_index


def el_nino_southern_oscillation_34(data_set, sst_name="sea-surface-temperature"):
    """Calculate the El Nino Southern Oscillation 3.4 index (ENSO 3.4)

    Following https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
    the index is derived from equatorial pacific sea-surface temperature (SST) anomalies in a box
    bordered by 5°S - 5°N and 170°W - 120°W. This translates to -5°N - 5°N and 190°E - 240°E.

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


def el_nino_southern_oscillation_4(data_set, sst_name="sea-surface-temperature"):
    """Calculate the El Nino Southern Oscillation 4 index (ENSO 4)

    Following https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
    the index is derived from equatorial pacific sea-surface temperature (SST) anomalies in a box
    bordered by 5°S - 5°N and 160°E - 150°W. This translates to -5°N - 5°N and 160°E - 210°E.

    Computation is done as follows:
    1. Compute area averaged total SST from Niño 4 region.
    2. Compute monthly climatology for area averaged total SST from Niño 4 region.
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
        Time series containing the ENSO 4 index.

    """
    sst_nino4 = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=-5,
        lat_north=5,
        lon_west=160,
        lon_east=210,
    )

    climatology = sst_nino4.groupby("time.month").mean("time")

    std_dev = sst_nino4.std("time")

    ENSO4_index = (sst_nino4.groupby("time.month") - climatology) / std_dev
    ENSO4_index = ENSO4_index.rename("ENSO4")

    return ENSO4_index


def tropical_north_atlantic_SST(data_set, sst_name="sea-surface-temperature"):
    """Calculate the sea-surface temperature (SST) anomaly index in the Tropical North Atlantic (SSTA_TNA)

    The Tropical North Atlantic region is defined by a box bordered by 5°N to 25°N and 55°W to 15°W.
    This translates to 5°N to 25°N and 305°E to 345°E.

    Computation is done as follows:
    1. Compute area averaged total SST in the region of interest.
    2. Compute monthly climatology for area averaged total SST from that region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.
    sst_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SSTA_TNA index.

    """
    sst = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=5,
        lat_north=25,
        lon_west=305,
        lon_east=345,
    )

    climatology = sst.groupby("time.month").mean("time")

    std_dev = sst.std("time")

    SSTA_TNA = (sst.groupby("time.month") - climatology) / std_dev
    SSTA_TNA = SSTA_TNA.rename("SSTA_TNA")

    return SSTA_TNA


def tropical_south_atlantic_SST(data_set, sst_name="sea-surface-temperature"):
    """Calculate the sea-surface temperature (SST) anomaly index in the Tropical South Atlantic (SSTA_TSA)

    The Tropical South Atlantic region is defined by a box bordered by 20°S to 0°N and 30°W to 10°E.
    This translates to -20°N to 0°N and 330°E to 10°E.

    Computation is done as follows:
    1. Compute area averaged total SST in the region of interest.
    2. Compute monthly climatology for area averaged total SST from that region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.
    sst_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SSTA_TSA index.

    """
    sst = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=-20,
        lat_north=0,
        lon_west=330,
        lon_east=10,
    )

    climatology = sst.groupby("time.month").mean("time")

    std_dev = sst.std("time")

    SSTA_TSA = (sst.groupby("time.month") - climatology) / std_dev
    SSTA_TSA = SSTA_TSA.rename("SSTA_TSA")

    return SSTA_TSA


def eastern_subtropical_indian_ocean_SST(data_set, sst_name="sea-surface-temperature"):
    """Calculate the sea-surface temperature (SST) anomaly index in the Eastern Subtropical Indian Ocean (SSTA_ESIO)

    The Eastern Subtropical Indian Ocean region is defined by a box bordered by 28°S to 18°S and 90°E to 100°E.
    This translates to -28°N to -18°N and 90°E to 100°E.

    Computation is done as follows:
    1. Compute area averaged total SST in the region of interest.
    2. Compute monthly climatology for area averaged total SST from that region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.
    sst_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SSTA_ESIO index.

    """
    sst = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=-28,
        lat_north=-18,
        lon_west=90,
        lon_east=100,
    )

    climatology = sst.groupby("time.month").mean("time")

    std_dev = sst.std("time")

    SSTA_ESIO = (sst.groupby("time.month") - climatology) / std_dev
    SSTA_ESIO = SSTA_ESIO.rename("SSTA_ESIO")

    return SSTA_ESIO


def western_subtropical_indian_ocean_SST(data_set, sst_name="sea-surface-temperature"):
    """Calculate the sea-surface temperature (SST) anomaly index in the Western Subtropical Indian Ocean (SSTA_WSIO)

    The Western Subtropical Indian Ocean region is defined by a box bordered by 37°S to 27°S and 55°E to 65°E.
    This translates to -37°N to -27°N and 55°E to 65°E.

    Computation is done as follows:
    1. Compute area averaged total SST in the region of interest.
    2. Compute monthly climatology for area averaged total SST from that region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.
    sst_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SSTA_WSIO index.

    """
    sst = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=-37,
        lat_north=-27,
        lon_west=55,
        lon_east=65,
    )

    climatology = sst.groupby("time.month").mean("time")

    std_dev = sst.std("time")

    SSTA_WSIO = (sst.groupby("time.month") - climatology) / std_dev
    SSTA_WSIO = SSTA_WSIO.rename("SSTA_WSIO")

    return SSTA_WSIO


def mediterranean_SST(data_set, sst_name="sea-surface-temperature"):
    """Calculate the sea-surface temperature (SST) anomaly index in the Mediterranean Sea (SSTA_MED)

    The Mediterranean Sea region is defined by a box bordered by 30°N to 45°N and 0°E to 25°E.

    Computation is done as follows:
    1. Compute area averaged total SST in the region of interest.
    2. Compute monthly climatology for area averaged total SST from that region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.
    sst_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SSTA_MED index.

    """
    sst = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=30,
        lat_north=45,
        lon_west=0,
        lon_east=25,
    )

    climatology = sst.groupby("time.month").mean("time")

    std_dev = sst.std("time")

    SSTA_MED = (sst.groupby("time.month") - climatology) / std_dev
    SSTA_MED = SSTA_MED.rename("SSTA_MED")

    return SSTA_MED


def hurricane_main_development_region_SST(data_set, sst_name="sea-surface-temperature"):
    """Calculate the sea-surface temperature (SST) anomaly index in Hurricane main development region (SSTA_HMDR)

    The Hurricane main development region is defined by a box bordered by 10°N to 20°N and 85°W to 20°W.
    This translates to 10°N to 20°N and 275°E to 340°E.

    Computation is done as follows:
    1. Compute area averaged total SST in the region of interest.
    2. Compute monthly climatology for area averaged total SST from that region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.
    sst_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SSTA_HMDR index.

    """
    sst = area_mean_weighted(
        dobj=data_set[sst_name],
        lat_south=10,
        lat_north=20,
        lon_west=275,
        lon_east=340,
    )

    climatology = sst.groupby("time.month").mean("time")

    std_dev = sst.std("time")

    SSTA_HMDR = (sst.groupby("time.month") - climatology) / std_dev
    SSTA_HMDR = SSTA_HMDR.rename("SSTA_HMDR")

    return SSTA_HMDR


def north_atlantic_sea_surface_salinity(data_set, sss_name="sea-surface-salinity"):
    """Calculate North Atlantic Sea-Surface Salinity index (NASSS)

    Following https://doi.org/10.1126/sciadv.1501588
    the index is derived from Atlantic sea-surface salinity (SSS) anomalies in a box
    spanning 50W-15W, 25N-50N.

    Note that in the original ref, there was a smaller box spanning 50W-40W, 37.5N-50N
    cut away from the one used here. We skip this here, for now.

    Computation is done as follows:
    1. Calculate the weighted area average of SSS in the region of interest.
    2. Compute monthly climatology for area averaged total SSS from that region.
    3. Subtract climatology from area averaged total SSS time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

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
    sss = area_mean_weighted(
        data_set[sss_name],
        lat_south=25,
        lat_north=50,
        lon_west=-50,
        lon_east=-15,
    )

    climatology = sss.groupby("time.month").mean("time")

    std_dev = sss.std("time")

    NASSS = (sss.groupby("time.month") - climatology) / std_dev
    NASSS = NASSS.rename("NASSS")

    return NASSS


def north_atlantic_sea_surface_salinity_west(data_set, sss_name="sea-surface-salinity"):
    """Calculate the Sea-Surface Salinity index in the Western part of the North Atlantic region (NASSS_W)

    Following https://doi.org/10.1126/sciadv.1501588
    the index is derived from Atlantic sea-surface salinity (SSS) anomalies. Here we focus on the Western part
    of the North Atlantic, defined by a box bordered by 25°N to 38°N and 50°W to 40°W.
    This translates to 25°N to 38°N and 310°E to 320°E.

    Computation is done as follows:
    1. Calculate the weighted area average of SSS in the region of interest.
    2. Compute monthly climatology for area averaged total SSS from that region.
    3. Subtract climatology from area averaged total SSS time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SSS field.
    sss_name: str
        Name of the Sea-Surface Salinity field. Defaults to "sea-surface-salinity".

    Returns
    -------
    xarray.DataArray
        Time series containing the NASSS_W index.

    """
    sss = area_mean_weighted(
        data_set[sss_name],
        lat_south=25,
        lat_north=38,
        lon_west=310,
        lon_east=320,
    )

    climatology = sss.groupby("time.month").mean("time")

    std_dev = sss.std("time")

    NASSS_W = (sss.groupby("time.month") - climatology) / std_dev
    NASSS_W = NASSS_W.rename("NASSS_W")

    return NASSS_W


def north_atlantic_sea_surface_salinity_east(data_set, sss_name="sea-surface-salinity"):
    """Calculate the Sea-Surface Salinity index in the Eastern part of the North Atlantic region (NASSS_E)

    Following https://doi.org/10.1126/sciadv.1501588
    the index is derived from Atlantic sea-surface salinity (SSS) anomalies. Here we focus on the Eastern part
    of the North Atlantic, defined by a box bordered by 25°N to 50°N and 40°W to 15°W.
    This translates to 25°N to 50°N and 320°E to 345°E.

    Computation is done as follows:
    1. Calculate the weighted area average of SSS in the region of interest.
    2. Compute monthly climatology for area averaged total SSS from that region.
    3. Subtract climatology from area averaged total SSS time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SSS field.
    sss_name: str
        Name of the Sea-Surface Salinity field. Defaults to "sea-surface-salinity".

    Returns
    -------
    xarray.DataArray
        Time series containing the NASSS_E index.

    """
    sss = area_mean_weighted(
        data_set[sss_name],
        lat_south=25,
        lat_north=50,
        lon_west=320,
        lon_east=345,
    )

    climatology = sss.groupby("time.month").mean("time")

    std_dev = sss.std("time")

    NASSS_E = (sss.groupby("time.month") - climatology) / std_dev
    NASSS_E = NASSS_E.rename("NASSS_E")

    return NASSS_E


def south_atlantic_sea_surface_salinity(data_set, sss_name="sea-surface-salinity"):
    """Calculate South Atlantic Sea-Surface Salinity index (SASSS)

    Following https://doi.org/10.1126/sciadv.1501588
    the index is derived from Atlantic sea-surface salinity (SSS) anomalies. Here we focus on
    the South Atlantic region, defined by a box bordered by 22.5°S to 10°S and 42°W to 10°W.
    This translates to -22.5°N to -10°N and 318°E to 350°E.

    Computation is done as follows:
    1. Calculate the weighted area average of SSS in the region of interest.
    2. Compute monthly climatology for area averaged total SSS from that region.
    3. Subtract climatology from area averaged total SSS time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SSS field.
    sss_name: str
        Name of the Sea-Surface Salinity field. Defaults to "sea-surface-salinity".

    Returns
    -------
    xarray.DataArray
        Time series containing the SASSS index.

    """
    sss = area_mean_weighted(
        data_set[sss_name],
        lat_south=-22.5,
        lat_north=-10,
        lon_west=318,
        lon_east=350,
    )

    climatology = sss.groupby("time.month").mean("time")

    std_dev = sss.std("time")

    SASSS = (sss.groupby("time.month") - climatology) / std_dev
    SASSS = SASSS.rename("SASSS")

    return SASSS


def atlantic_multidecadal_oscillation(data_set, sst_name="sea-surface-temperature"):
    """Calculate the Atlantic Multi-decadal Oscillation (AMO) index.

    This follows the NOAA method <https://psl.noaa.gov/data/timeseries/AMO/> in defining the Atlantic Multi-decadal Oscillation
    index using area weighted averaged sea-surface temperature anomalies of the north Atlantic between 0°N and 70°N,
    The anomalies are relative to a monthly climatology calculated from the whole time covered by the data set.
    It differs from the definition of the NOAA in that it does not detrend the time series and the smomothing is not performed.

    Computation is done as follows:
    1. Compute area averaged total SST from north Atlantic region.
    2. Compute monthly climatology for area averaged total SST from north Atlantic  region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.

    Further informations can be found in :
    - [Trenberth and Shea, 2006] <https://doi.org/10.1029/2006GL026894>.
    - NCAR climate data guide <https://climatedataguide.ucar.edu/climate-data/atlantic-multi-decadal-oscillation-amo>


    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SST field.
    slp_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-tempearture".

    Returns
    -------
    xarray.DataArray
        Time series containing the AMO index.

    """
    sst = data_set[sst_name]

    # create Atlantic polygon and calculate horizontal average
    atlanctic_polygon_lon_lat = polygon_prime_meridian(
        Polygon([(15, 0), (-65, 0), (-105, 25), (-45, 70), (15, 70), (-7, 35)])
    )
    sst_box_ave = area_mean_weighted_polygon_selection(
        dobj=sst, polygon_lon_lat=atlanctic_polygon_lon_lat
    )

    AMO = monthly_anomalies_unweighted(sst_box_ave)

    AMO = AMO.rename("AMO")

    return AMO


def sea_air_surface_temperature_anomaly_north_all(
    data_set, sat_name="sea-air-temperature"
):
    """Calculate the Sea Air Surface Temperature Anomaly (SASTA) index, for the complete northern hemisphere.

    Land and Ocean data is used for the calculation. The anomalies are relative to a monthly climatology
    calculated from the whole time covered by the data set.

    Computation is done as follows:
    1. Compute area averaged total SAT for the hemisphere.
    2. Compute monthly climatology for area averaged SAT for the hemisphere.
    3. Subtract climatology from area averaged total SAT time series to obtain anomalies.
    This follows https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies

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
        Name of the DataArray will be: "SASTAI-north-all"

    """
    sat = data_set[sat_name]
    sat_mean = area_mean_weighted(
        dobj=sat, lat_south=0, lat_north=90, lon_west=0, lon_east=360
    )

    SASTAI = monthly_anomalies_unweighted(sat_mean)
    SASTAI = SASTAI.rename("SASTAI-north-all")

    return SASTAI


def sea_air_surface_temperature_anomaly_north_ocean(
    data_set, sat_name="sea-air-temperature"
):
    """Calculate the Sea Air Surface Temperature Anomaly (SASTA) index, for the northern hemisphere Ocean.

    Only data over the Ocean is used for the calculation. The anomalies are relative to a monthly climatology
    calculated from the whole time covered by the data set.

    Computation is done as follows:
    1. Compute area averaged total SAT for the hemisphere.
    2. Compute monthly climatology for area averaged SAT for the hemisphere.
    3. Subtract climatology from area averaged total SAT time series to obtain anomalies.
    This follows https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies

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
        Name of the DataArray will be: "SASTAI-north-ocean"

    """
    # select only ocean data
    sat = data_set[sat_name].where(data_set["is_over_ocean"])
    sat_mean = area_mean_weighted(
        dobj=sat, lat_south=0, lat_north=90, lon_west=0, lon_east=360
    )

    SASTAI = monthly_anomalies_unweighted(sat_mean)
    SASTAI = SASTAI.rename("SASTAI-north-ocean")

    return SASTAI


def sea_air_surface_temperature_anomaly_north_land(
    data_set, sat_name="sea-air-temperature"
):
    """Calculate the Sea Air Surface Temperature Anomaly (SASTA) index, for the northern hemisphere land.

    Only data over land is used for the calculation. The anomalies are relative to a monthly climatology
    calculated from the whole time covered by the data set.

    Computation is done as follows:
    1. Compute area averaged total SAT for the hemisphere.
    2. Compute monthly climatology for area averaged SAT for the hemisphere.
    3. Subtract climatology from area averaged total SAT time series to obtain anomalies.
    This follows https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies

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
        Name of the DataArray will be: "SASTAI-north-land"

    """
    # select only land data
    sat = data_set[sat_name].where(~data_set["is_over_ocean"])
    sat_mean = area_mean_weighted(
        dobj=sat, lat_south=0, lat_north=90, lon_west=0, lon_east=360
    )

    SASTAI = monthly_anomalies_unweighted(sat_mean)
    SASTAI = SASTAI.rename("SASTAI-north-land")

    return SASTAI


def sea_air_surface_temperature_anomaly_south_all(
    data_set, sat_name="sea-air-temperature"
):
    """Calculate the Sea Air Surface Temperature Anomaly (SASTA) index, for the complete southern hemisphere.

    Land and Ocean data is used for the calculation. The anomalies are relative to a monthly climatology
    calculated from the whole time covered by the data set.

    Computation is done as follows:
    1. Compute area averaged total SAT for the hemisphere.
    2. Compute monthly climatology for area averaged SAT for the hemisphere.
    3. Subtract climatology from area averaged total SAT time series to obtain anomalies.
    This follows https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SAT field.
    slp_name: str
        Name of the Sea-Air Temperature field. Defaults to "sea-air-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SASTA index for southern hemisphere.
        Name of the DataArray will be: "SASTAI-south-all"

    """
    sat = data_set[sat_name]
    sat_mean = area_mean_weighted(
        dobj=sat, lat_south=-90, lat_north=0, lon_west=0, lon_east=360
    )

    SASTAI = monthly_anomalies_unweighted(sat_mean)
    SASTAI = SASTAI.rename("SASTAI-south-all")

    return SASTAI


def sea_air_surface_temperature_anomaly_south_ocean(
    data_set, sat_name="sea-air-temperature"
):
    """Calculate the Sea Air Surface Temperature Anomaly (SASTA) index, for the southern hemisphere Ocean.

    Only data over the Ocean is used for the calculation. The anomalies are relative to a monthly climatology
    calculated from the whole time covered by the data set.

    Computation is done as follows:
    1. Compute area averaged total SAT for the hemisphere.
    2. Compute monthly climatology for area averaged SAT for the hemisphere.
    3. Subtract climatology from area averaged total SAT time series to obtain anomalies.
    This follows https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SAT field.
    slp_name: str
        Name of the Sea-Air Temperature field. Defaults to "sea-air-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SASTA index for southern hemisphere.
        Name of the DataArray will be: "SASTAI-south-ocean"

    """
    # select only ocean data
    sat = data_set[sat_name].where(data_set["is_over_ocean"])
    sat_mean = area_mean_weighted(
        dobj=sat, lat_south=-90, lat_north=0, lon_west=0, lon_east=360
    )

    SASTAI = monthly_anomalies_unweighted(sat_mean)
    SASTAI = SASTAI.rename("SASTAI-south-ocean")

    return SASTAI


def sea_air_surface_temperature_anomaly_south_land(
    data_set, sat_name="sea-air-temperature"
):
    """Calculate the Sea Air Surface Temperature Anomaly (SASTA) index, for the southern hemisphere land.

    Only data over land is used for the calculation. The anomalies are relative to a monthly climatology
    calculated from the whole time covered by the data set.

    Computation is done as follows:
    1. Compute area averaged total SAT for the hemisphere.
    2. Compute monthly climatology for area averaged SAT for the hemisphere.
    3. Subtract climatology from area averaged total SAT time series to obtain anomalies.
    This follows https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SAT field.
    slp_name: str
        Name of the Sea-Air Temperature field. Defaults to "sea-air-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the SASTA index for southern hemisphere.
        Name of the DataArray will be: "SASTAI-south-land"

    """
    # select only land data
    sat = data_set[sat_name].where(~data_set["is_over_ocean"])
    sat_mean = area_mean_weighted(
        dobj=sat, lat_south=-90, lat_north=0, lon_west=0, lon_east=360
    )

    SASTAI = monthly_anomalies_unweighted(sat_mean)
    SASTAI = SASTAI.rename("SASTAI-south-land")

    return SASTAI


def sahel_precipitation_anomaly(data_set, precip_name="precipitation"):
    """Calculate the Sahel precipitation anomaly index

    Following http://research.jisao.washington.edu/data/sahel/
    the Sahel rainy season is centered on June through October. The Sahel precipitation index in its original form gives a measure
    for year to year variability of Sahel rainfall as mean over this rainy season.

    Here we compute the Sahel precipitation anomaly index as monthly anomalies of rainfall in the Sahel zone.
    As defined in https://doi.org/10.1175/JAMC-D-13-0181.1
    the Sahel zone is assumed to be bordered by 10 to 20°N and 20°W to 10°E.

    Computation is done as follows:
    1. Compute area averaged total precipitation from Sahel zone.
    2. Compute monthly climatology for area averaged total precipitation from Sahel zone.
    3. Subtract climatology from area averaged total precipitation time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.


    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an precipitation field.
    precip_name: str
        Name of the precipitation field. Defaults to "precipitation".

    Returns
    -------
    xarray.DataArray
        Time series containing the Sahel precipitation anomaly index.

    """
    precip = area_mean_weighted(
        dobj=data_set[precip_name],
        lat_south=10,
        lat_north=20,
        lon_west=340,
        lon_east=10,
    )

    climatology = precip.groupby("time.month").mean("time")

    std_dev = precip.std("time")

    Sahel_precip = (precip.groupby("time.month") - climatology) / std_dev
    Sahel_precip = Sahel_precip.rename("SPAI")

    return Sahel_precip


def pacific_decadal_oscillation_pc(data_set, sst_name="sea-surface-temperature"):
    """Calculate the principal component based Pacific Decadal Oscillation (PDO) index

    Following
    https://climatedataguide.ucar.edu/climate-data/pacific-decadal-oscillation-pdo-definition-and-indices
    this index is obtained as Principle Component (PC) time series of the leading Empirical Orthogonal Function (EOF)
    of monthly sea-surface temperature (SST) anomalies in the North Pacific basin, 20°-60°N, 120°-260°E.

    Note: To have the EOFs truly orthogonal, we need to take the area of the grid cells into account.
    For equidistant latitude/longitude grids the area weights are proportional to cos(latitude).
    Before applying Singular Value Decomposition, input data is multiplied with the square root of the weights.

    Computation is done as follows:
    1. Compute sea-surface temperature anomalies in North Pacific basin.
    2. Flatten spatial dimensions and subtract mean in time.
    3. Perform Singular Value Decomposition.
    4. Normalize Principal Components.
    5. Obtain PDO index as PC time series related to leading EOF.
    6. Restore leading EOF pattern.
    7. Use negative pole of leading EOF to correct index sign - if necessary.


    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.
    sst_name: str
        Name of the Sea-Surface Temperature field. Defaults to "sea-surface-temperature".

    Returns
    -------
    xarray.DataArray
        Time series containing the PDO index.

    """

    mask = (
        (data_set.coords["lat"] >= 20)
        & (data_set.coords["lat"] <= 60)
        & (data_set.coords["lon"] >= 120)
        & (data_set.coords["lon"] <= 260)
    )

    sst = data_set[sst_name]
    sst = sst * eof_weights(sst)
    sst = sst.where(mask)
    climatology = sst.groupby("time.month").mean("time")
    sst = (sst.groupby("time.month") - climatology).drop("month")

    sst_flat = sst.stack(tmp_space=("lat", "lon")).dropna(dim="tmp_space")

    pc, s, eof = sp.linalg.svd(sst_flat - sst_flat.mean(axis=0), full_matrices=False)

    pc_std = pc.std(axis=0)
    pc /= pc_std

    PDO_index = xr.DataArray(pc[:, 0], dims=("time"), coords={"time": sst["time"]})

    eofs = sst.stack(tmp_space=("lat", "lon")).copy()[0:1, :]
    eofs[0:1, eofs[0].notnull().values] = (
        eof * pc_std[:, np.newaxis] * s[:, np.newaxis]
    )[0:1, :]
    eofs = eofs.unstack(dim="tmp_space").rename(**{"time": "mode"})

    mask_neg = (
        (eofs[0].coords["lat"] >= 30)
        & (eofs[0].coords["lat"] <= 50)
        & (eofs[0].coords["lon"] >= 150)
        & (eofs[0].coords["lon"] <= 200)
    )

    if eofs[0].where(mask_neg).mean(("lat", "lon")) > 0:
        PDO_index.values = -PDO_index.values

    PDO_index = PDO_index.rename("PDO_PC")

    return PDO_index


def north_pacific(data_set, slp_name="sea-level-pressure"):
    """Calculate the North Pacific index (NP)

    Following https://climatedataguide.ucar.edu/climate-data/north-pacific-np-index-trenberth-and-hurrell-monthly-and-winter
    the index is derived from area-weighted sea level pressure (SLP) anomalies in a box
    bordered by 30°N to 65°N and 160°E to 140°W. This translates to 30°N to 65°N and 160°E to 220°E.

    Computation is done as follows:
    1. Compute area averaged total SLP from region of interest.
    2. Compute monthly climatology for area averaged total SLP from that region.
    3. Subtract climatology from area averaged total SLP time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Note: Usually the index focusses on anomalies during November and March. Here we keep full information and
    compute monthly anomalies for all months of a year.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SLP field.
    sst_name: str
        Name of the sea level pressure field. Defaults to "sea--level-pressure".

    Returns
    -------
    xarray.DataArray
        Time series containing the North Pacific index.

    """
    slp = area_mean_weighted(
        dobj=data_set[slp_name],
        lat_south=30,
        lat_north=65,
        lon_west=160,
        lon_east=220,
    )

    climatology = slp.groupby("time.month").mean("time")

    std_dev = slp.std("time")

    NP_index = (slp.groupby("time.month") - climatology) / std_dev
    NP_index = NP_index.rename("NP")

    return NP_index


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
    southern_oscillation = partial(southern_oscillation)
    north_atlantic_oscillation = partial(north_atlantic_oscillation)
    north_atlantic_oscillation_pc = partial(north_atlantic_oscillation_pc)
    el_nino_southern_oscillation_12 = partial(el_nino_southern_oscillation_12)
    el_nino_southern_oscillation_3 = partial(el_nino_southern_oscillation_3)
    el_nino_southern_oscillation_34 = partial(el_nino_southern_oscillation_34)
    el_nino_southern_oscillation_4 = partial(el_nino_southern_oscillation_4)
    tropical_north_atlantic_SST = partial(tropical_north_atlantic_SST)
    tropical_south_atlantic_SST = partial(tropical_south_atlantic_SST)
    eastern_subtropical_indian_ocean_SST = partial(eastern_subtropical_indian_ocean_SST)
    western_subtropical_indian_ocean_SST = partial(western_subtropical_indian_ocean_SST)
    mediterranean_SST = partial(mediterranean_SST)
    hurricane_main_development_region_SST = partial(
        hurricane_main_development_region_SST
    )
    north_atlantic_sea_surface_salinity = partial(north_atlantic_sea_surface_salinity)
    north_atlantic_sea_surface_salinity_west = partial(
        north_atlantic_sea_surface_salinity_west
    )
    north_atlantic_sea_surface_salinity_east = partial(
        north_atlantic_sea_surface_salinity_east
    )
    south_atlantic_sea_surface_salinity = partial(south_atlantic_sea_surface_salinity)
    atlantic_multidecadal_oscillation = partial(atlantic_multidecadal_oscillation)
    sea_air_surface_temperature_anomaly_north_all = partial(
        sea_air_surface_temperature_anomaly_north_all
    )
    sea_air_surface_temperature_anomaly_north_ocean = partial(
        sea_air_surface_temperature_anomaly_north_ocean
    )
    sea_air_surface_temperature_anomaly_north_land = partial(
        sea_air_surface_temperature_anomaly_north_land
    )
    sea_air_surface_temperature_anomaly_south_all = partial(
        sea_air_surface_temperature_anomaly_south_all
    )
    sea_air_surface_temperature_anomaly_south_ocean = partial(
        sea_air_surface_temperature_anomaly_south_ocean
    )
    sea_air_surface_temperature_anomaly_south_land = partial(
        sea_air_surface_temperature_anomaly_south_land
    )
    sahel_precipitation_anomaly = partial(sahel_precipitation_anomaly)
    pacific_decadal_oscillation_pc = partial(pacific_decadal_oscillation_pc)
    north_pacific = partial(north_pacific)

    @classmethod
    def get_all_names(cls):
        """Return tuple with all known index names."""
        return tuple(cls.__members__.keys())
