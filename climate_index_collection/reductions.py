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
        Stddev data. Has the same variable name(s) as dobj.
    
    """
    mean_of_squares = mean_weighted(dobj ** 2, weights=weights, dim=dim)
    square_of_means = mean_weighted(dobj, weights=weights, dim=dim) ** 2
    variance = mean_of_squares - square_of_means
    return variance


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
    std_dev = variance ** 0.5
    return std_dev

