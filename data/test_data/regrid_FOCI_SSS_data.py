from pathlib import Path

import numpy as np
import xarray as xr

from xhistogram.xarray import histogram as xhist


# find input files and create output file names
original_data_files = [
    "FOCI/FOCI1.3-SW038_1m_23500101_23541231_grid_T.nc",
    "FOCI/FOCI1.3-SW038_1m_23500101_33491231_grid_T.nc",
]
regridded_data_files = [
    "FOCI/FOCI1.3-SW038_1m_23500101_23541231_grid_T_atmos_grid.nc",
    "FOCI/FOCI1.3-SW038_1m_23500101_33491231_grid_T_atmos_grid.nc",
]
target_grid_files = [
    "FOCI/FOCI1.3-SW038_echam6_ATM_mm_2350-2359_geopoth_pl_monthly_50000.nc",
    "FOCI/FOCI1.3-SW038_echam6_ATM_mm_2350-3349_geopoth_pl_monthly_50000.nc",
]

for infile, outfile, gridfile in zip(
    original_data_files,
    regridded_data_files,
    target_grid_files,
):
    if not (Path(infile).exists() and Path(gridfile).exists()):
        print(f"Cannot work on {infile}. Skipping.")
        break

    # open input and fix redundancy in x (halo layers) and y (folding)
    original_data_set = xr.open_dataset(infile, chunks={"time_counter": 30})
    original_data_set = original_data_set.isel(x=slice(None, -2), y=slice(None, -1))

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
        [np.inf]
        + list(
            (
                target_grid.lat.isel(lat=slice(0, -1)).data
                + target_grid.lat.isel(lat=slice(1, None)).data
            )
            / 2.0
        )
        + [-np.inf]
    )
    # shift the orig lon so it exactly fits the bins
    source_lon = original_data_set.nav_lon
    source_lon = source_lon % 360
    source_lon = xr.where(source_lon < lon_bins[-1], source_lon, source_lon - 360)

    # binning
    numerator = xhist(
        original_data_set.nav_lat,
        source_lon,
        bins=[lat_bins[::-1], lon_bins],
        weights=original_data_set.sosaline,
        dim=("x", "y"),
    )
    denominator = xhist(
        original_data_set.nav_lat,
        source_lon,
        bins=[lat_bins[::-1], lon_bins],
        weights=(original_data_set.sosaline != 0).astype(float),
        dim=("x", "y"),
    )
    binned = numerator / denominator

    # fix names and coord values in binned dataset
    binned = binned.rename({"nav_lon_bin": "lon"})
    binned.coords["lon"] = target_grid.lon

    binned = binned.rename({"nav_lat_bin": "lat"})
    binned = binned.isel(lat=slice(None, None, -1))
    binned.coords["lat"] = target_grid.lat
    binned.attrs.update(original_data_set.sosaline.attrs)

    binned = binned.rename("sosaline").to_dataset()

    binned = binned.rename({"time_counter": "time"})
    binned.coords["time"] = target_grid.time.isel(time=slice(0, binned.dims["time"]))

    # compute and save
    binned = binned.compute()
    binned.to_netcdf(outfile)
