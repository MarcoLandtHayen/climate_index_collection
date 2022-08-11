from enum import Enum
from functools import partial

import numpy as np

from .reductions import area_mean_weighted, mean_unweighted


def southern_annular_mode(data_set, slp_name="sea-level-pressure"):
    """Calculate the southern annular mode index.

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
    north_atlantic_oscillation = partial(north_atlantic_oscillation)
    el_nino_southern_oscillation_34 = partial(el_nino_southern_oscillation_34)

    @classmethod
    def get_all_names(cls):
        """Return tuple with all known index names."""
        return tuple(cls.__members__.keys())
