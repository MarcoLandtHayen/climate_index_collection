import numpy as np
import pandas as pd
import xarray as xr


def _is_start_of_next_month(timestamp):
    return (timestamp.day == 1) & (timestamp.hour == 0)


def _get_prev_month(year=None, month=None):
    if month == 1:
        return year - 1, 12
    else:
        return year, month - 1


def _get_next_month(year=None, month=None):
    if month == 12:
        return year + 1, 1
    else:
        return year, month + 1


def _get_fixed_year_month(timestamp):
    # if possible, convert to datetime with Pandas
    # (this won't be necessary for cftime types)
    try:
        timestamp = pd.to_datetime(timestamp)
    except Exception as e:
        pass
    if _is_start_of_next_month(timestamp):
        return _get_prev_month(year=timestamp.year, month=timestamp.month)
    else:
        return timestamp.year, timestamp.month


def _get_mid_of_month(year=None, month=None):
    year_upper, month_upper = _get_next_month(year=year, month=month)
    lower = np.datetime64(f"{year:04d}-{month:02d}-01T00:00:00")
    upper = np.datetime64(f"{year_upper:04d}-{month_upper:02d}-01T00:00:00")
    return lower + (upper - lower) / 2


def fix_monthly_time_stamps(dobj, time_name="time"):
    """Fix monthly time stamps to be exactly the middle of months.
    
    There's a few input data sets which label monthly data at the turn of months.
    Let's unify the convention to mid month everywhere.
    
    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Should contain a time dimension.
    time_name: str
        Defaults to "time".

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Same as dobj but with fixed time axis.
    
    """
    # extract time axis
    orig_time = dobj.coords[time_name]

    # construct new time axis
    fixed_time = [
        _get_mid_of_month(*_get_fixed_year_month(ts)) for ts in orig_time.data
    ]

    # assign coord
    fixed_dobj = dobj.copy()
    fixed_dobj = fixed_dobj.assign_coords(
        {time_name: xr.DataArray(fixed_time, dims=(time_name,))}
    )

    return fixed_dobj
