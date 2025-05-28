#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs

def get_healpix_info(ds):
    """
    Extract HEALPix parameters from a dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset
        
    Returns:
    --------
    tuple
        (nside, nest, npix)
    """
    # Convert DataArray to Dataset if needed
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    
    # Get HEALPix parameters from crs attributes
    if 'crs' in ds.variables and hasattr(ds.crs, 'healpix_nside'):
        nside = ds.crs.healpix_nside
        nest = ds.crs.healpix_order == 'nest'
        npix = 12 * nside * nside
    else:
        # Try to calculate nside from the cell count
        npix = ds.dims.get('cell')
        if npix:
            nside = hp.npix2nside(npix)
            # Default to nested ordering as that's common
            nest = True
        else:
            raise ValueError("Could not determine HEALPix parameters from dataset")
    
    return nside, nest, npix

def add_latlon_coords(ds):
    """
    Add latitude and longitude coordinates to a HEALPix dataset if they don't exist.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset
        
    Returns:
    --------
    xarray.Dataset
        Dataset with latitude and longitude coordinates
    """
    # Convert DataArray to Dataset if needed
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    
    if 'lat' in ds.coords and 'lon' in ds.coords:
        return ds
    
    # Get HEALPix parameters
    nside, nest, npix = get_healpix_info(ds)
    
    # Create pixel indices for all cells in the dataset
    pixel_indices = np.arange(ds.dims['cell'])
    
    # Get latitude and longitude for each pixel
    theta, phi = hp.pix2ang(nside, pixel_indices, nest=nest)
    
    # Convert to lat/lon in degrees
    lat = 90 - np.degrees(theta)  # Latitude: 90° at north pole, -90° at south pole
    lon = np.degrees(phi)         # Longitude: 0° to 360°
    
    # Adjust longitude to be in range [-180, 180]
    lon = np.where(lon > 180, lon - 360, lon)
    
    # Add coordinates to the dataset
    ds = ds.assign_coords(lat=('cell', lat), lon=('cell', lon))
    
    return ds

def select_at_latitude(ds, latitude, tolerance=1.0):
    """
    Select data points along a specific latitude with a given tolerance.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    latitude : float
        Target latitude in degrees
    tolerance : float, optional
        Tolerance in degrees for latitude selection, default is 1.0
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only points at the specified latitude (within tolerance)
    """
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Select points within the latitude band
    mask = (ds.lat >= latitude - tolerance) & (ds.lat <= latitude + tolerance)
    result = ds.where(mask, drop=True)
    
    # Return as original type
    if is_dataarray:
        return result[original_name]
    return result

def select_at_longitude(ds, longitude, tolerance=1.0):
    """
    Select data points along a specific longitude with a given tolerance.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    longitude : float
        Target longitude in degrees (-180 to 180)
    tolerance : float, optional
        Tolerance in degrees for longitude selection, default is 1.0
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only points at the specified longitude (within tolerance)
    """
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Normalize longitude to -180 to 180 range
    longitude = ((longitude + 180) % 360) - 180
    
    # Handle longitude wrapping (e.g., -180 and 180 are the same)
    # Calculate the minimum difference considering the wrap-around
    lon_diff = np.minimum(
        np.abs(ds.lon - longitude),
        np.minimum(
            np.abs(ds.lon - (longitude + 360)),
            np.abs(ds.lon - (longitude - 360))
        )
    )
    
    # Select points within the longitude band
    mask = lon_diff <= tolerance
    result = ds.where(mask, drop=True)
    
    # Return as original type
    if is_dataarray:
        return result[original_name]
    return result

def select_at_points(ds, points):
    """
    Select data at specific latitude/longitude points using nearest neighbor.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    points : list of tuples or array-like
        List of (latitude, longitude) pairs
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only the nearest points to the specified coordinates
    """
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Get HEALPix parameters
    nside, nest, _ = get_healpix_info(ds)
    
    # Convert lat/lon points to pixel indices
    points = np.array(points)
    
    # Normalize longitudes to be in range [0, 360)
    lons = points[:, 1] % 360
    
    # Convert from lat/lon to theta/phi (healpy uses co-latitude)
    theta = np.radians(90 - points[:, 0])  # Convert latitude to co-latitude in radians
    phi = np.radians(lons)                 # Convert longitude to radians
    
    # Get the pixel indices
    pixels = hp.ang2pix(nside, theta, phi, nest=nest)
    
    # Select the data at those pixels
    result = ds.isel(cell=pixels)
    
    # Return as original type
    if is_dataarray:
        return result[original_name]
    return result

def select_region(ds, lat_min, lat_max, lon_min, lon_max):
    """
    Select data within a rectangular lat/lon region.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    lat_min, lat_max : float
        Minimum and maximum latitude in degrees
    lon_min, lon_max : float
        Minimum and maximum longitude in degrees (-180 to 180)
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only points within the specified region
    """
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Normalize longitude range
    lon_min = ((lon_min + 180) % 360) - 180
    lon_max = ((lon_max + 180) % 360) - 180
    
    # Create mask for the region
    lat_mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
    
    # Handle longitude wrapping
    if lon_min <= lon_max:
        lon_mask = (ds.lon >= lon_min) & (ds.lon <= lon_max)
    else:
        # Crosses the dateline
        lon_mask = (ds.lon >= lon_min) | (ds.lon <= lon_max)
    
    # Combine masks
    mask = lat_mask & lon_mask
    
    result = ds.where(mask, drop=True)
    
    # Return as original type
    if is_dataarray:
        return result[original_name]
    return result

def interpolate_to_grid(ds, var_name=None, lat_res=1.0, lon_res=1.0, 
                        lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
                        method='nearest'):
    """
    Interpolate a HEALPix dataset to a regular lat/lon grid.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    var_name : str, optional
        Name of the variable to interpolate (only needed if ds is a Dataset)
    lat_res, lon_res : float, optional
        Resolution of the output grid in degrees
    lat_min, lat_max, lon_min, lon_max : float, optional
        Boundaries of the output grid
    method : str, optional
        Interpolation method: 'nearest', 'linear', etc.
        
    Returns:
    --------
    xarray.DataArray
        Regular grid data array
    """
    # Handle DataArray input
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        data_array = ds
        var_name = ds.name
        ds = ds.to_dataset()
    elif var_name is None:
        # If ds is a Dataset and var_name is not provided, use the first data variable
        var_name = list(ds.data_vars)[0]
        data_array = ds[var_name]
    else:
        data_array = ds[var_name]
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Create target grid
    lats = np.arange(lat_min, lat_max + lat_res, lat_res)
    lons = np.arange(lon_min, lon_max + lon_res, lon_res)
    
    # Create output grid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    output = np.zeros(lat_grid.shape)
    
    # Get HEALPix parameters
    nside, nest, _ = get_healpix_info(ds)
    
    # For each point in the output grid, find the nearest HEALPix pixel
    for i in range(len(lats)):
        for j in range(len(lons)):
            lat = lats[i]
            lon = lons[j]
            
            # Convert lat/lon to HEALPix pixel index
            theta = np.radians(90 - lat)  # Convert latitude to co-latitude
            phi = np.radians(lon % 360)   # Convert longitude to radians
            pixel = hp.ang2pix(nside, theta, phi, nest=nest)
            
            # Get the value at that pixel
            output[i, j] = data_array.isel(cell=pixel).values.item()
    
    # Create a DataArray with the regular grid
    result = xr.DataArray(
        data=output,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        name=var_name
    )
    
    return result


def plot_healpix_selection(data, title=None, cmap='inferno', projection='PlateCarree'):
    """
    Plot HEALPix data with proper coordinate handling.
    
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        HEALPix dataset or data array with lat/lon coordinates
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name
    projection : str, optional
        Map projection name from cartopy.crs
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Ensure we have a DataArray
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) == 0:
            raise ValueError("Dataset has no data variables")
        # Use the first data variable
        var_name = list(data.data_vars)[0]
        data = data[var_name]
    
    # Ensure lat/lon coordinates exist
    if 'lat' not in data.coords or 'lon' not in data.coords:
        raise ValueError("Data must have lat/lon coordinates. Use add_latlon_coords first.")
    
    # Create figure with map projection
    proj = getattr(ccrs, projection)()
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': proj})
    
    # Plot the data
    scatter = ax.scatter(
        data.lon, data.lat, 
        c=data.values, 
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        s=10,  # Point size
        alpha=0.7  # Transparency
    )
    
    # Add coastlines and grid
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label=data.name if data.name else 'Value')
    
    # Set title
    if title:
        plt.title(title)
    
    return fig


def get_value_at_latlon(ds, lat, lon, method='nearest'):
    """
    Get the value at a specific latitude/longitude point.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    method : str, optional
        Interpolation method: 'nearest', 'linear', etc.
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray or float
        Value at the specified point
    """
    # Handle DataArray input
    is_dataarray = isinstance(ds, xr.DataArray)
    original_name = None
    
    if is_dataarray:
        original_name = ds.name
        ds_with_coords = add_latlon_coords(ds.to_dataset())
    else:
        ds_with_coords = add_latlon_coords(ds)
    
    # For nearest neighbor method
    if method == 'nearest':
        # Get HEALPix parameters
        nside, nest, _ = get_healpix_info(ds_with_coords)
        
        # Convert lat/lon to pixel index
        theta = np.radians(90 - lat)  # Convert latitude to co-latitude in radians
        phi = np.radians(lon % 360)   # Convert longitude to radians
        
        # Get the pixel index
        pixel = hp.ang2pix(nside, theta, phi, nest=nest)
        
        # Return the value at that pixel
        if is_dataarray:
            return ds_with_coords[original_name].isel(cell=pixel).values.item()
        else:
            return ds_with_coords.isel(cell=pixel)
    
    # For other interpolation methods
    else:
        # Create a single-point dataset for interpolation
        point = xr.Dataset(coords={'lat': [lat], 'lon': [lon]})
        
        # Interpolate
        if is_dataarray:
            result = ds_with_coords[original_name].interp(lat=point.lat, lon=point.lon, method=method)
            return result.values.item()
        else:
            return ds_with_coords.interp(lat=point.lat, lon=point.lon, method=method)
