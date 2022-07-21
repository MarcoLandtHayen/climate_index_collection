import numpy as np 

def latitude_longitude_specs(dobj, minimal_feedback = True) :
    """This function will indicate what kind of longitude dimensions are used in a dataset.

    Parameters
    ----------
    dobj: xarray.DataArray
        Contains the dimension to check.
    minimal_feedback : bool
        If set to False, the returned dataset will include further details.

    Returns
    -------
    lon_dict: dict
        Dictonary containing information about:
        minimum : 
        maximum : 
        diff : 
        diff_is_constant :
        diff_is_positive : 
        diff_changes :
        # based on the idea by kith: https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy
    lat_dict: dict
        as lon_dict but for latitudes 
    """
    def get_properties(data) :

        # calculate minimum and maximum based on numpy.
        minimum = np.min(data)
        maximum = np.max(data)
        # calculate the diff between each dimension value.
        diff = np.diff(data)
        # the diff is constant, if all diff values are the same as first value. 
        diff_is_constant = np.all(diff == diff[0])
        # if the difference is constant, directly calculate the sign of change
        if diff_is_constant:
            diff = diff[0]
            diff_change_values = None
            diff_change_position = None
            diff_sign = diff > 0
        else :
            pos = np.where(diff[:-1] != diff[1:])[0]
            if np.size(pos) == np.size(diff) -1 :
                diff_change_values  = 'all'
                diff_change_position = 'all'
            else :
                print(pos)
                diff_change_values = diff[pos]
                diff_change_position = pos
            if np.all(diff > 0) :
                diff_sign = True
            elif np.all(diff < 0) :
                diff_sign = False
            else :
                diff_
        
        if minimal_feedback :
            if not isinstance(diff, float) :
                diff = None
            return dict(
                mini = minimum,
                mini_pos = np.where(data == minimum)[0][0],
                maxi = np.max(data),
                maxi_pos = np.where(data == maximum)[0][0],
                size = np.size(data), 
                diff_constant = diff_is_constant,
                diff_sign = diff_sign,
                diff = diff,
                )
        else : 
            return dict(
                mini = minimum,
                mini_pos = np.where(data == minimum)[0][0],
                maxi = np.max(data),
                maxi_pos = np.where(data == maximum)[0][0],
                size = np.size(data), 
                diff_constant = diff_is_constant,
                diff_sign = diff_sign,
                diff = diff,
                diff_change_position = diff_change_position,
                diff_change_values = diff_change_values
                )
    # results 
    return dict(
        lat = get_properties(dobj.lat.values), 
        lon = get_properties(dobj.lon.values)
        )
