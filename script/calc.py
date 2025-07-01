#!/usr/bin/env python
"""
fgCO2 Climate Index Correlation Analysis
========================================

This script calculates point-by-point Pearson correlations between fgCO2 field data
and climate indices (DMI and ONI) for annual and seasonal periods.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: June 30, 2025
License: WTFPL

Output:
-------
Single NetCDF file containing correlation coefficients, p-values, and significance
flags for both DMI and ONI indices across Annual, DJF, MAM, JJA, and SON periods.
"""

import xarray as xr
import numpy as np
from scipy import stats
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


def ensure_output_directory(filepath):
    """
    Ensure the output directory exists.
    
    Parameters
    ----------
    filepath : str
        Path to the output file
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def align_time_series(data1, data2, name1="Dataset1", name2="Dataset2"):
    """
    Align two time series to their common time period.
    
    Parameters
    ----------
    data1 : xr.DataArray
        First time series
    data2 : xr.DataArray
        Second time series
    name1 : str
        Name of first dataset for logging
    name2 : str
        Name of second dataset for logging
    
    Returns
    -------
    tuple
        Aligned data arrays and common time range
    """
    # Ensure time coordinates are datetime
    data1['time'] = pd.to_datetime(data1.time.values)
    data2['time'] = pd.to_datetime(data2.time.values)
    
    # Find common time period
    start_time = max(data1.time.min().values, data2.time.min().values)
    end_time = min(data1.time.max().values, data2.time.max().values)
    
    print(f"  {name1} time range: {data1.time.min().values} to {data1.time.max().values}")
    print(f"  {name2} time range: {data2.time.min().values} to {data2.time.max().values}")
    print(f"  Common period: {pd.Timestamp(start_time).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(end_time).strftime('%Y-%m-%d')}")
    
    # Subset to common period
    data1_aligned = data1.sel(time=slice(start_time, end_time))
    data2_aligned = data2.sel(time=slice(start_time, end_time))
    
    return data1_aligned, data2_aligned, (start_time, end_time)


def calculate_period_mean(data, period, seasons):
    """
    Calculate temporal mean for a specific period (Annual or seasonal).
    
    Parameters
    ----------
    data : xr.DataArray
        Input data array
    period : str
        Period name ('Annual', 'DJF', 'MAM', 'JJA', 'SON')
    seasons : dict
        Dictionary mapping season names to month lists
    
    Returns
    -------
    xr.DataArray
        Period-averaged data
    """
    if period == 'Annual':
        # Annual mean
        return data.resample(time='AS').mean()
    else:
        # Seasonal mean
        season_months = seasons[period]
        
        # Select months for this season
        season_data = data.where(data.time.dt.month.isin(season_months), drop=True)
        
        # Group by year and calculate seasonal means
        return season_data.groupby('time.year').mean()


def calculate_correlations(fgco2_period, index_period, lat_dims, lon_dims):
    """
    Calculate grid-point correlations between fgCO2 and climate index.
    
    Parameters
    ----------
    fgco2_period : xr.DataArray
        Period-averaged fgCO2 data
    index_period : xr.DataArray
        Period-averaged climate index data
    lat_dims : xr.DataArray
        Latitude coordinates
    lon_dims : xr.DataArray
        Longitude coordinates
    
    Returns
    -------
    tuple
        Arrays of correlation coefficients, p-values, and significance flags
    """
    n_lat = len(lat_dims)
    n_lon = len(lon_dims)
    
    # Initialize output arrays
    corr_coef = np.full((n_lat, n_lon), np.nan)
    p_values = np.full((n_lat, n_lon), np.nan)
    significance = np.zeros((n_lat, n_lon), dtype=bool)
    
    # Align time dimensions
    if hasattr(fgco2_period, 'year'):
        # For seasonal data grouped by year
        common_years = np.intersect1d(fgco2_period.year.values, index_period.year.values)
        fgco2_aligned = fgco2_period.sel(year=common_years)
        index_aligned = index_period.sel(year=common_years).values
    else:
        # For annual resampled data
        min_len = min(len(fgco2_period.time), len(index_period.time))
        fgco2_aligned = fgco2_period.isel(time=slice(0, min_len))
        index_aligned = index_period.isel(time=slice(0, min_len)).values
    
    # Calculate correlation for each grid point
    for i in range(n_lat):
        for j in range(n_lon):
            try:
                # Extract time series for this grid point
                fgco2_point = fgco2_aligned.isel(latitude=i, longitude=j).values
                
                # Remove NaN values
                mask = ~(np.isnan(fgco2_point) | np.isnan(index_aligned))
                
                if np.sum(mask) >= 3:  # Minimum 3 points for meaningful correlation
                    # Calculate Pearson correlation
                    r, p = stats.pearsonr(fgco2_point[mask], index_aligned[mask])
                    
                    corr_coef[i, j] = r
                    p_values[i, j] = p
                    significance[i, j] = p < 0.05
            except Exception as e:
                # Keep NaN values if calculation fails
                continue
    
    return corr_coef, p_values, significance


def process_climate_index(fgco2_data, index_data, index_name, periods, seasons, lat_dims, lon_dims):
    """
    Process correlations for a single climate index across all periods.
    
    Parameters
    ----------
    fgco2_data : xr.DataArray
        fgCO2 data aligned to common time period
    index_data : xr.DataArray
        Climate index data aligned to common time period
    index_name : str
        Name of the climate index ('DMI' or 'ONI')
    periods : list
        List of period names
    seasons : dict
        Dictionary mapping season names to month lists
    lat_dims : xr.DataArray
        Latitude coordinates
    lon_dims : xr.DataArray
        Longitude coordinates
    
    Returns
    -------
    dict
        Dictionary containing correlation results for all periods
    """
    print(f"\nProcessing {index_name} correlations...")
    
    results = {
        'correlation': [],
        'p_value': [],
        'significant': []
    }
    
    for period in periods:
        print(f"  Calculating {period} correlations...")
        
        # Calculate period means
        fgco2_period = calculate_period_mean(fgco2_data, period, seasons)
        index_period = calculate_period_mean(index_data, period, seasons)
        
        # Calculate correlations
        corr, pval, sig = calculate_correlations(fgco2_period, index_period, lat_dims, lon_dims)
        
        results['correlation'].append(corr)
        results['p_value'].append(pval)
        results['significant'].append(sig)
        
        # Print period statistics
        valid_mask = ~np.isnan(corr)
        n_valid = np.sum(valid_mask)
        if n_valid > 0:
            print(f"    Valid points: {n_valid}/{corr.size} ({100*n_valid/corr.size:.1f}%)")
            print(f"    Mean r: {np.nanmean(corr):.3f}, Range: [{np.nanmin(corr):.3f}, {np.nanmax(corr):.3f}]")
            print(f"    Significant: {np.sum(sig)} points ({100*np.sum(sig)/n_valid:.1f}% of valid)")
    
    # Stack results
    for key in results:
        results[key] = np.stack(results[key])
    
    return results


def create_output_dataset(dmi_results, oni_results, periods, lat_dims, lon_dims, 
                         time_ranges, creator_info):
    """
    Create the output NetCDF dataset with all results.
    
    Parameters
    ----------
    dmi_results : dict
        DMI correlation results
    oni_results : dict
        ONI correlation results
    periods : list
        List of period names
    lat_dims : xr.DataArray
        Latitude coordinates
    lon_dims : xr.DataArray
        Longitude coordinates
    time_ranges : dict
        Common time ranges for each index
    creator_info : dict
        Creator metadata
    
    Returns
    -------
    xr.Dataset
        Complete output dataset
    """
    # Create dimension coordinates
    period_coord = xr.DataArray(periods, dims=['period'])
    
    # Create the dataset
    ds = xr.Dataset({
        # DMI variables
        'dmi_correlation': xr.DataArray(
            dmi_results['correlation'],
            dims=['period', 'latitude', 'longitude'],
            coords={
                'period': period_coord,
                'latitude': lat_dims,
                'longitude': lon_dims
            },
            attrs={
                'long_name': 'Pearson correlation coefficient between fgCO2 and DMI',
                'standard_name': 'correlation_coefficient',
                'units': '1',
                'valid_range': [-1.0, 1.0]
            }
        ),
        'dmi_p_value': xr.DataArray(
            dmi_results['p_value'],
            dims=['period', 'latitude', 'longitude'],
            coords={
                'period': period_coord,
                'latitude': lat_dims,
                'longitude': lon_dims
            },
            attrs={
                'long_name': 'Two-tailed p-value for DMI correlation test',
                'units': '1',
                'valid_range': [0.0, 1.0]
            }
        ),
        'dmi_significant': xr.DataArray(
            dmi_results['significant'],
            dims=['period', 'latitude', 'longitude'],
            coords={
                'period': period_coord,
                'latitude': lat_dims,
                'longitude': lon_dims
            },
            attrs={
                'long_name': 'DMI correlation significance at 95% confidence level',
                'units': '1',
                'flag_values': [0, 1],
                'flag_meanings': 'not_significant significant',
                'description': 'True (1) if p-value < 0.05, False (0) otherwise'
            }
        ),
        # ONI variables
        'oni_correlation': xr.DataArray(
            oni_results['correlation'],
            dims=['period', 'latitude', 'longitude'],
            coords={
                'period': period_coord,
                'latitude': lat_dims,
                'longitude': lon_dims
            },
            attrs={
                'long_name': 'Pearson correlation coefficient between fgCO2 and ONI',
                'standard_name': 'correlation_coefficient',
                'units': '1',
                'valid_range': [-1.0, 1.0]
            }
        ),
        'oni_p_value': xr.DataArray(
            oni_results['p_value'],
            dims=['period', 'latitude', 'longitude'],
            coords={
                'period': period_coord,
                'latitude': lat_dims,
                'longitude': lon_dims
            },
            attrs={
                'long_name': 'Two-tailed p-value for ONI correlation test',
                'units': '1',
                'valid_range': [0.0, 1.0]
            }
        ),
        'oni_significant': xr.DataArray(
            oni_results['significant'],
            dims=['period', 'latitude', 'longitude'],
            coords={
                'period': period_coord,
                'latitude': lat_dims,
                'longitude': lon_dims
            },
            attrs={
                'long_name': 'ONI correlation significance at 95% confidence level',
                'units': '1',
                'flag_values': [0, 1],
                'flag_meanings': 'not_significant significant',
                'description': 'True (1) if p-value < 0.05, False (0) otherwise'
            }
        )
    })
    
    # Add coordinate attributes
    ds.period.attrs.update({
        'long_name': 'Analysis period',
        'standard_name': 'time_period',
        'description': 'Annual and seasonal averaging periods'
    })
    
    ds.latitude.attrs.update({
        'long_name': 'Latitude',
        'standard_name': 'latitude',
        'units': 'degrees_north',
        'axis': 'Y'
    })
    
    ds.longitude.attrs.update({
        'long_name': 'Longitude',
        'standard_name': 'longitude',
        'units': 'degrees_east',
        'axis': 'X'
    })
    
    # Add global attributes
    ds.attrs.update({
        'title': 'fgCO2 Climate Index Correlation Analysis',
        'institution': 'University of California, Riverside',
        'source': 'Correlation analysis between fgCO2 and climate indices (DMI, ONI)',
        'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'references': 'Pearson correlation analysis with two-tailed significance test',
        'creator_name': creator_info['name'],
        'creator_email': creator_info['email'],
        'Conventions': 'CF-1.8',
        'project': 'fgCO2 Climate Teleconnections Analysis',
        'processing_level': 'Level 3 - Derived geophysical variables',
        'period_definitions': ('Annual: Calendar year average; '
                             'DJF: December-January-February average; '
                             'MAM: March-April-May average; '
                             'JJA: June-July-August average; '
                             'SON: September-October-November average'),
        'statistical_significance': 'p < 0.05 (95% confidence level)',
        'dmi_time_range': f"{pd.Timestamp(time_ranges['DMI'][0]).strftime('%Y-%m-%d')} to "
                          f"{pd.Timestamp(time_ranges['DMI'][1]).strftime('%Y-%m-%d')}",
        'oni_time_range': f"{pd.Timestamp(time_ranges['ONI'][0]).strftime('%Y-%m-%d')} to "
                          f"{pd.Timestamp(time_ranges['ONI'][1]).strftime('%Y-%m-%d')}"
    })
    
    return ds


def main():
    """
    Main execution function for fgCO2 climate index correlation analysis.
    """
    print("=" * 80)
    print("fgCO2 Climate Index Correlation Analysis")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    config = {
        'paths': {
            'fgco2': "../raw_data/fgco2_1998_2024.nc",
            'dmi': "../raw_data/dmi.nc",
            'oni': "../raw_data/oni.nc",
            'output': "../processed_data/fgco2_climate_indices_correlations.nc"
        },
        'creator': {
            'name': 'Sandy Herho',
            'email': 'sandy.herho@email.ucr.edu'
        },
        'periods': ['Annual', 'DJF', 'MAM', 'JJA', 'SON'],
        'seasons': {
            'DJF': [12, 1, 2],
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11]
        }
    }
    
    # Ensure output directory exists
    ensure_output_directory(config['paths']['output'])
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        fgco2 = xr.open_dataset(config['paths']['fgco2'])
        dmi = xr.open_dataset(config['paths']['dmi'])
        oni = xr.open_dataset(config['paths']['oni'])
        print("  All datasets loaded successfully")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Extract data variables
    fgco2_data = fgco2['fgco2']
    dmi_data = dmi['value']
    oni_data = oni['value']
    
    # Get spatial dimensions
    lat_dims = fgco2_data.latitude
    lon_dims = fgco2_data.longitude
    print(f"  Spatial grid: {len(lat_dims)} x {len(lon_dims)} "
          f"(latitude x longitude)")
    
    # Dictionary to store time ranges
    time_ranges = {}
    
    # Process DMI correlations
    print("\nDMI Analysis:")
    print("-" * 40)
    fgco2_dmi, dmi_aligned, time_ranges['DMI'] = align_time_series(
        fgco2_data, dmi_data, "fgCO2", "DMI"
    )
    
    dmi_results = process_climate_index(
        fgco2_dmi, dmi_aligned, 'DMI', 
        config['periods'], config['seasons'], 
        lat_dims, lon_dims
    )
    
    # Process ONI correlations
    print("\nONI Analysis:")
    print("-" * 40)
    fgco2_oni, oni_aligned, time_ranges['ONI'] = align_time_series(
        fgco2_data, oni_data, "fgCO2", "ONI"
    )
    
    oni_results = process_climate_index(
        fgco2_oni, oni_aligned, 'ONI', 
        config['periods'], config['seasons'], 
        lat_dims, lon_dims
    )
    
    # Create output dataset
    print("\nCreating output dataset...")
    output_ds = create_output_dataset(
        dmi_results, oni_results, 
        config['periods'], lat_dims, lon_dims,
        time_ranges, config['creator']
    )
    
    # Save to NetCDF
    print(f"\nSaving results to: {config['paths']['output']}")
    encoding = {var: {'zlib': True, 'complevel': 4} for var in output_ds.data_vars}
    output_ds.to_netcdf(config['paths']['output'], encoding=encoding)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print(f"Output file: {config['paths']['output']}")
    print(f"File size: {os.path.getsize(config['paths']['output']) / 1024 / 1024:.1f} MB")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
