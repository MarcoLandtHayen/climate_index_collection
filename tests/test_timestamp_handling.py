import xarray as xr
import cftime
import numpy as np
import datetime


from climate_index_collection.timestamp_handling import fix_monthly_time_stamps


def test_time_stamp_fixing_to_mid_month():
    """Test that end-of-month labels are moved to mid months"""

    # construct time axes
    wrong_time_axis = [
        cftime.DatetimeProlepticGregorian(2022, 1, 31, 23, 58, 0),
        cftime.DatetimeProlepticGregorian(2022, 2, 15, 12, 0, 0),  # that's not mid Feb!
        cftime.DatetimeProlepticGregorian(2022, 4, 1, 0, 0, 0),  # that's end of March!
    ]
    desired_time_axis = [
        # note we can't do `(date1 + date2) / 2` but `date1 + (date2 - date1) / 2` works
        cftime.DatetimeProlepticGregorian(2022, 1, 1, 0, 0, 0)
        + (
            cftime.DatetimeProlepticGregorian(2022, 2, 1, 0, 0, 0)
            - cftime.DatetimeProlepticGregorian(2022, 1, 1, 0, 0, 0)
        )
        / 2,
        cftime.DatetimeProlepticGregorian(2022, 2, 1, 0, 0, 0)
        + (
            cftime.DatetimeProlepticGregorian(2022, 3, 1, 0, 0, 0)
            - cftime.DatetimeProlepticGregorian(2022, 2, 1, 0, 0, 0)
        )
        / 2,
        cftime.DatetimeProlepticGregorian(2022, 3, 1, 0, 0, 0)
        + (
            cftime.DatetimeProlepticGregorian(2022, 4, 1, 0, 0, 0)
            - cftime.DatetimeProlepticGregorian(2022, 3, 1, 0, 0, 0)
        )
        / 2,
    ]

    # put into data array
    dobj = xr.DataArray(
        [1, 2, 3], name="dummy", dims=("time",), coords={"time": wrong_time_axis}
    )

    # get fixed array
    fixed_dobj = fix_monthly_time_stamps(dobj, time_name="time")

    # check closeness
    np.testing.assert_almost_equal(
        (fixed_dobj.time.data - desired_time_axis) / datetime.timedelta(seconds=1),
        desired=0,
        decimal=3,
    )

