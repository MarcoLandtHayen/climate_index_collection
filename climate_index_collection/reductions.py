import numpy as np


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



def monthly_weights(dobj, normalize = True):
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
    dim = 'time'
    groupby_dim = "time.month"
    month_length = dobj.time.dt.days_in_month
    num_groups = len(np.unique(dobj.time.dt.month))
    
    # Check if weights shall be normalized
    if normalize :
        weights = month_length.groupby(groupby_dim) / month_length.groupby(groupby_dim).sum('time')
        # Test that the sum of the weights for each season is 1.0
        np.testing.assert_allclose(weights.groupby(groupby_dim).sum().values, np.ones(num_groups))
    else :
        weights = month_length
    
    return weights


def monthly_mean_weighted(dobj):
    """Calculates the weighted monthly mean values of a dataset.
    It will make use of the grouped_mean_weighted function, which is similar to the mean_weigthed function, 
    but additionally allow to include a dimension to group the data by.
    It takes care of leap years and thus differs from "monthly_mean_unweighted"
    Adapted from: https://docs.xarray.dev/en/stable/examples/monthly-means.html
    
    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Monthly mean data. Has the same variable name(s) as dobj. 
        Dimension 'time' will be removed.
        Dimension 'month' is gained. 
            Int values, starting with 1 for January and ending with 12 for December.
    """
    # The following setting will allow to calculate the monthly mean weighted
    dim = 'time'
    groupby_dim = "time.month"
    weights = monthly_weights(dobj)
    
    return grouped_mean_weighted(dobj = dobj, weights = weights, dim = dim, groupby_dim= groupby_dim)


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
    return mean_unweighted(dobj = dobj.groupby("time.month"), dim = 'time')


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
    return (dobj.groupby('time.month') - monthly_mean_unweighted(dobj)).drop_vars('month') 


def monthly_anomalies_weighted(dobj):
    """Calculates the weighted monthly anomalies from the weighted monthly climatology of a dataset.
    The weighted monthly climatology is calculated using "monthly_mean_weighted"
    Takes care of leap years and thus differs from "monthly_anomalies"

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Weighted monthly anomalies from the weighted monthly climatology of the original data. 
        Has the same variable name(s) as dobj.
    """
    # Note: the dobj needs to be grouped by the same group_dim as the DataArray returned by monthly_mean_weighted!
    return (dobj.groupby('time.month') - monthly_mean_weighted(dobj)).drop_vars('month')