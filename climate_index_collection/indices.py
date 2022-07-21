from enum import Enum
from functools import partial

from .reductions import mean_unweighted, mean_weighted, monthly_anomalies_weighted
from .data_specs import latitude_longitude_specs

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
    """Calculate the area based El Nino Southern Oscillation 3.4 index (ENSO 3.4)

    This uses equatorial pacific sea-surface temperature anomalies in a box 
    borderd by 5°S, 5°N, 120°W and 170°W.
    !!!!!!
    Further describtion needed here
    !!!!!!
 
    The procedure is as follows:
    1. A spacial mean is calculated for the SST inside the box described above
    2. The monthly mean is removed from the previous result
    
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
    LatBounds = (-5,5)      #°N which are 5°S and 5°N
    LonBounds = (190, 250)  #°E which are 170°W and 120°W
    
    LatLondSpecs = latitude_longitude_specs(dobj = data_set)
    
    # check if the latitude is stricktly increasing or decreasing
    if LatLondSpecs['lat']['diff_sign'] == True : 
        LatSlice = slice(LatBounds[0], LatBounds[1])
    elif LatLondSpecs['lat']['diff_sign'] == False :
        LatSlice = slice(LatBounds[1], LatBounds[0])
    # if the slope changes sign, this needs to be corrected
    elif LatLondSpecs['lat']['diff_sign'] == None :
        raise Exception('It seems the Latitude is not strictly increasing or decreasing \nNeed fix this!')

    # check if the longitude is stricktly increasing or decreasing

    if LatLondSpecs['lon']['diff_sign'] == True : 
        LonSlice = slice(LonBounds[0], LonBounds[1])
    elif LatLondSpecs['lon']['diff_sign'] == False :
        LonSlice = slice(LonBounds[1], LonBounds[0])
    elif LatLondSpecs['lon']['diff_sign'] == None :
        raise Exception('It seems the Longitude is not strictly increasing or decreasing \nNeed fix this!')
        
    sst = data_set[sst_name].sel(lat = LatSlice, lon = LonSlice)
    sst_mean = mean_unweighted(dobj = sst, 
                                dim = {'lat', 'lon'})
    ENSO_index = monthly_anomalies_weighted(sst_mean) 
    return ENSO_index


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
