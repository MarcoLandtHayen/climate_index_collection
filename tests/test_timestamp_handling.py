import numpy as np
import xarray as xr


from climate_index_collection.timestamp_handling import fix_monthly_time_stamps


def test_time_stamp_fixing_to_mid_month():
    """Test that end-of-month labels are moved to mid months"""

    # construct time axes
    wrong_time_axis = [
        np.datetime64("2022-01-31T23:58:00", "ns"),
        np.datetime64("2022-02-15T00:00:00", "ns"),  # that's not exactly mid Feb!
        np.datetime64("2022-04-01T00:00:00", "ns"),  # that's end of March!
    ]
    desired_time_axis = [
        # note we can't do `(date1 + date2) / 2` but `date1 + (date2 - date1) / 2` works
        np.datetime64("2022-01-01T00:00:00", "ns")
        + (
            np.datetime64("2022-02-01T00:00:00", "ns")
            - np.datetime64("2022-01-01T00:00:00", "ns")
        )
        / 2,
        np.datetime64("2022-02-01T00:00:00", "ns")
        + (
            np.datetime64("2022-03-01T00:00:00", "ns")
            - np.datetime64("2022-02-01T00:00:00", "ns")
        )
        / 2,
        np.datetime64("2022-03-01T00:00:00", "ns")
        + (
            np.datetime64("2022-04-01T00:00:00", "ns")
            - np.datetime64("2022-03-01T00:00:00", "ns")
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
        (fixed_dobj.time.data - desired_time_axis) / np.timedelta64(1, "s"),
        desired=0,
        decimal=3,
    )

