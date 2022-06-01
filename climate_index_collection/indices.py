def southern_annular_mode(data_set, slp_name="sea-level-pressure"):
    """Calculate the southern annular mode index.
    
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
    
    slp40S = slp.sel(lat=-40, method="nearest").mean("lon")
    slp65S = slp.sel(lat=-65, method="nearest").mean("lon")
    
    slp_diff = (slp40S - slp65S)
    
    SAM_index = (slp_diff - slp_diff.mean("time")) / slp_diff.std("time")
    SAM_index = SAM_index.rename("SAM")
    
    return SAM_index