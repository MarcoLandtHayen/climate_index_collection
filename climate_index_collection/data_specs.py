import numpy as np 

def get_spacial_dimension_specs(data) :
    """
    This function returns informations about the structure of a dimension given in data.
    Those are e.g. 
    - Are the difference between the grid lines constant or not. 
    - Are all difference positive or negative 
    For more details look into Returns.
    Data can either be latitude or longitude values.
    
    NOTE: The difference is defined for latitude and longitude in the same manner:
        diff = np.diff(data)
        - data is a np.array() containing values of latitude or longitude
        - diff is a np.array() containing the differnce betẃeen neighboring values of l
    
    # Parts are based on the idea by kith: https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy
    
    Parameters
    ----------
    data: np.array()
        Contains the dimension for which informations shall be given.

    Returns
    -------
    dict :
        This containes the following keys:
        minimum : float 
            Minumum value of data
        minimum_position : int
            Position where the minumum is found
        maximum : float
            Maximum value of data
        maximum_position : int
            Position where the maximum is found
        size : int
            length of data
        diff_is_constant : bool
            True if difference between data values is constant, else False
        diff_is_positive : bool or None
            True:   if all differences are positive, 
            False:  if all are negative,
            None:   if mixed signs occure              
        diff_change_postition : bool or list
            False:  if differences are constant (diff_is_constant is True), 
            True:   if differences are changing betweeen all postitions in l,
            List:   else, containing all positions in the diff array where the following value is different from the value at position n
        diff_change_values : bool or list
            Analog to diff_change_position.
            List:   containing all vallues at the position given by diff_change_position
        diff : np.array()
            List containing 
    
    Example:
    -------    
    data = [0,1,2,4,6,8,11,12,13]
    diff = [1,1,2,2,2,3,1,1] will result in 
    dict(
        mini = 0,
        mini_pos = 0,
        maxi = 13,
        maxi_pos = 8,
        size = 9, 
        diff_constant = False,
        diff_sign = True,
        diff = [1,1,2,2,2,3,1,1],
        diff_change_position = [0,2,5,6],
        diff_change_values = [1,2,3,1]
        )
    """
    # calculate minimum and maximum based on numpy.
    minimum = np.min(data)
    maximum = np.max(data)
    # calculate the diff between each dimension value.
    diff = np.diff(data)
    # the diff is constant, if all diff values are the same as first value. 
    diff_is_constant = np.all(diff == diff[0])

    # If the difference is constant, diff is set to the first value. 
    # Directly calculate the sign of change
    # And set other parameters to False 
    if diff_is_constant:
        diff = diff[0]
        diff_change_values = False
        diff_change_position = False
        diff_sign = diff > 0
    # Else, calulate the positions where diff changes values.
    else :
        pos = np.where(diff[:-1] != diff[1:])[0]
        pos = np.array(pos)
        # if change in every position:
        if np.size(pos) == np.size(diff) -1 :
            diff_change_values  = True
            diff_change_position = True
        # otherwise set values and position to corresponding keywords
        # diff_change_values and diff_change_position
        else :
            diff_change_values = diff[pos]
            diff_change_position = pos
        # check if all signs are either positive or negative or mixed 
        # and set corresponding values
        if np.all(diff > 0) :
            diff_sign = True
        elif np.all(diff < 0) :
            diff_sign = False
        else :
            diff_sign = None

    return dict(
            mini = minimum,
            mini_pos = np.where(data == minimum)[0][0],
            maxi = np.max(data),
            maxi_pos = np.where(data == maximum)[0][0],
            size = np.size(data), 
            diff_constant = diff_is_constant,
            diff_sign = diff_sign,
            diff_change_position = diff_change_position,
            diff_change_values = diff_change_values,
            diff = diff,
            )

def latitude_longitude_specs(dobj, lat_name ='lat', lon_name='lon') :
    """This function will return information about the latitude and longitude 
    structure of the data.
    The information for both are calculated with get_spacial_dimension_specs.
    
    Parameters
    ----------
    dobj: xarray.DataArray
        Data for which the latitude and longitude informations shall be given.
    lat_name: str
        Name of the latitude dimension. Defaults to "lat".
    lon_name: str
        Name of the longitude dimension. Defaults to "lon".
    Returns
    -------
    dict with keys:
        lat_specs : dict containing specifics for latitude dimension
        lon_specs : dict containing specifics for longitude dimension
    """

    # rcalculate the specifications for latitude and longitude and write into dictonary
    return dict(
        lat_specs = get_spacial_dimension_specs(dobj[lat_name].values), 
        lon_specs = get_spacial_dimension_specs(dobj[lon_name].values)
        )

def sel_latitude_longitude_slice(dobj, LatBounds = None, LonBounds = None, lat_name = 'lat', lon_name = 'lon'):
    """Returns latitude and longitude slices for given latitude and longitude bounds 
    dependent on the data sets latitude and longitude structure.
    The informations given about increasing or decreasing latitude and longitude values 
    are derived by the latitude_longitude_specs function.
    
    Parameters
    ----------
    dobj: xarray.DataArray
        Data for which the latitude and longitude informations shall be given.
    LatBounds : list, tupel or np.array
        Latitude Bounds for which the correct slice shall be obtained.
        Only max and min values will be used ! 
    LonBounds : list, tupel or np.array
        Longitude Bounds for which the correct slice shall be obtained.
        Only max and min values will be used !
    lat_name: str
        Name of the latitude dimension. Defaults to "lat".
    lon_name: str
        Name of the longitude dimension. Defaults to "lon".

    Returns
    -------
    slice :
        Latitude slice
    slice : 
        Longitude slice
    """
    
    LatLondSpecs = latitude_longitude_specs(dobj = dobj)
    
    # check if the latitude is stricktly increasing or decreasing
    if LatLondSpecs['lat_specs']['diff_sign'] == True : 
        LatSlice = slice(np.min(LatBounds), np.max(LatBounds))
    elif LatLondSpecs['lat_specs']['diff_sign'] == False :
        LatSlice = slice(np.max(LatBounds), np.min(LatBounds))
    # if the slope changes sign, this needs to be corrected
    elif LatLondSpecs['lat_specs']['diff_sign'] == None :
        raise Exception('It seems the Latitude is not strictly increasing or decreasing \nNeed fix this!')

    # check if the longitude is stricktly increasing or decreasing

    if LatLondSpecs['lon_specs']['diff_sign'] == True : 
        LonSlice = slice(np.min(LonBounds), np.max(LonBounds))
    elif LatLondSpecs['lon_specs']['diff_sign'] == False :
        LonSlice = slice(np.max(LonBounds), np.min(LonBounds))
    elif LatLondSpecs['lon_specs']['diff_sign'] == None :
        raise Exception('It seems the Longitude is not strictly increasing or decreasing \nNeed fix this!')
    
    return LatSlice, LonSlice