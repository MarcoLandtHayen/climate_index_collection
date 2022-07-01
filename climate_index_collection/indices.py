from enum import Enum
from functools import partial

from .reductions import mean_unweighted


def southern_annular_mode(data_set, slp_name="sea-level-pressure"):
    """Calculate the southern annular mode index.

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
    """Calculate the North Atlantic Oscillation (NAO) index station-based using grid points closest to
    Reykjavik (64°9'N, 21°56'W) and Ponta Delgada (37°45'N, 25°40'W).

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

    slp_diff = slp_northern_station - slp_southern_station

    NAO_index = slp_northern_station_norm - slp_southern_station_norm
    NAO_index = NAO_index.rename("NAO")

    return NAO_index


class ClimateIndexFunctions(Enum):
    """Enumeration of all index functions.

    We can use this to, e.g., iterate over all defined indices.

    >>> for index_func in ClimateIndexFunctions:
    ...     print(index_func.name)
    southern_annular_mode
    ...

    """

    southern_annular_mode = partial(southern_annular_mode)

    @classmethod
    def get_all_names(cls):
        """Return tuple with all known index names."""
        return tuple(cls.__members__.keys())
