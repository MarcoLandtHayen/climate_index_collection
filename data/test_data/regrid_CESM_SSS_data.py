from pathlib import Path

import numpy as np
import xarray as xr

from xhistogram.xarray import histogram as xhist


# find input files and create output file names
original_data_files = [
    "CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.pop.h.0001-0005.SSS.nc",
    "CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.pop.h.0001-0999.SSS.nc",
]
regridded_data_files = [
    "CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.pop.h.0001-0005.SSS.atmos_grid.nc",
    "CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.pop.h.0001-0999.SSS.atmos_grid.nc",
]
target_grid_files = [
    "CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0010.SST.nc",
    "CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0999.SST.nc",
]

for infile, outfile, gridfile in zip(
    original_data_files,
    regridded_data_files,
    target_grid_files,
):
    if not (Path(infile).exists() and Path(gridfile).exists()):
        print(f"Cannot work on {infile}. Skipping.")
        break

    # open input data
    original_data_set = xr.open_dataset(infile)

    # get target grid
    target_grid = xr.open_dataset(gridfile)
    target_grid = target_grid[["time", "lat", "lon"]]

    # construct bin edges
    dlon = (target_grid.lon).diff("lon").mean().data[()]  # assume equidistant
    lon_bins = np.arange(
        target_grid.lon.isel(lon=0) - dlon / 2,
        target_grid.lon.isel(lon=-1) + 3 * dlon / 2,
        dlon,
    )

    lat_bins = np.array(
        [-np.inf]
        + list(
            (
                target_grid.lat.isel(lat=slice(0, -1)).data
                + target_grid.lat.isel(lat=slice(1, None)).data
            )
            / 2.0
        )
        + [np.inf]
    )

    # shift the orig lon so it exactly fits the bins
    source_lon = original_data_set.TLONG
    source_lon = source_lon % 360
    source_lon = xr.where(source_lon < lon_bins[-1], source_lon, source_lon - 360)

    # binning
    numerator = xhist(
        original_data_set.TLAT,
        source_lon,
        bins=[lat_bins, lon_bins],
        weights=original_data_set.SALT,
        dim=("nlon", "nlat"),
    )
    denominator = xhist(
        original_data_set.TLAT,
        source_lon,
        bins=[lat_bins, lon_bins],
        weights=(original_data_set.SALT != 0).astype(float),
        dim=("nlon", "nlat"),
    )
    binned = numerator / denominator

    # fix names and coord values in binned dataset
    binned = binned.rename({"TLONG_bin": "lon"})
    binned.coords["lon"] = target_grid.lon

    binned = binned.rename({"TLAT_bin": "lat"})
    binned = binned.isel(lat=slice(None, None, -1))
    binned.coords["lat"] = target_grid.lat
    binned.attrs.update(original_data_set.SALT.attrs)

    binned = binned.rename("SALT").to_dataset()

    binned.coords["time"] = target_grid.time.isel(time=slice(0, binned.dims["time"]))

    # compute and save
    binned = binned.compute()
    binned.to_netcdf(outfile)
