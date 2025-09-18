#!/usr/bin/env python
"""
fgCO2 Climate Index Multi-metric Analysis
=========================================

This script calculates point-by-point Pearson correlation, Spearman correlation,
and Mutual Information between fgCO2 field data and climate indices (DMI and ONI) 
for annual and seasonal periods.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: September 17, 2025
License: WTFPL

Output:
-------
- NetCDF file containing three metrics (Pearson, Spearman, MI), p-values, and 
  significance flags for both DMI and ONI indices across all periods
- Text statistics report in ../stats directory
"""

import xarray as xr
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


def ensure_directory(filepath):
    """Ensure the output directory exists."""
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


def calculate_mutual_information(x, y, n_bins=10):
    """
    Calculate mutual information between two continuous variables.
    
    Parameters
    ----------
    x : array
        First variable
    y : array
        Second variable
    n_bins : int
        Number of bins for discretization
    
    Returns
    -------
    float
        Mutual information score (normalized 0-1)
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 3:
        return np.nan
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    try:
        # Use mutual information regression for continuous variables
        # Reshape for sklearn
        mi_score = mutual_info_regression(
            x_clean.reshape(-1, 1), 
            y_clean, 
            n_neighbors=3,
            random_state=42
        )[0]
        
        # Normalize to 0-1 range (approximate)
        # Max MI is roughly log(n) where n is number of samples
        max_mi = np.log(len(x_clean))
        normalized_mi = min(mi_score / max_mi, 1.0) if max_mi > 0 else 0
        
        return normalized_mi
        
    except Exception:
        return np.nan


def calculate_multi_metrics(fgco2_period, index_period, lat_dims, lon_dims):
    """
    Calculate multiple metrics between fgCO2 and climate index.
    
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
    dict
        Dictionary containing all metrics and significance tests
    """
    n_lat = len(lat_dims)
    n_lon = len(lon_dims)
    
    # Initialize output arrays for all metrics
    results = {
        'pearson_corr': np.full((n_lat, n_lon), np.nan),
        'pearson_pval': np.full((n_lat, n_lon), np.nan),
        'pearson_sig': np.zeros((n_lat, n_lon), dtype=bool),
        'spearman_corr': np.full((n_lat, n_lon), np.nan),
        'spearman_pval': np.full((n_lat, n_lon), np.nan),
        'spearman_sig': np.zeros((n_lat, n_lon), dtype=bool),
        'mutual_info': np.full((n_lat, n_lon), np.nan),
        'mutual_info_sig': np.zeros((n_lat, n_lon), dtype=bool)  # MI > 0.1 threshold
    }
    
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
    
    # Calculate metrics for each grid point
    for i in range(n_lat):
        for j in range(n_lon):
            try:
                # Extract time series for this grid point
                fgco2_point = fgco2_aligned.isel(latitude=i, longitude=j).values
                
                # Remove NaN values
                mask = ~(np.isnan(fgco2_point) | np.isnan(index_aligned))
                
                if np.sum(mask) >= 3:  # Minimum 3 points for meaningful correlation
                    clean_fgco2 = fgco2_point[mask]
                    clean_index = index_aligned[mask]
                    
                    # Pearson correlation
                    r_pearson, p_pearson = stats.pearsonr(clean_fgco2, clean_index)
                    results['pearson_corr'][i, j] = r_pearson
                    results['pearson_pval'][i, j] = p_pearson
                    results['pearson_sig'][i, j] = p_pearson < 0.05
                    
                    # Spearman correlation
                    r_spearman, p_spearman = stats.spearmanr(clean_fgco2, clean_index)
                    results['spearman_corr'][i, j] = r_spearman
                    results['spearman_pval'][i, j] = p_spearman
                    results['spearman_sig'][i, j] = p_spearman < 0.05
                    
                    # Mutual Information
                    mi = calculate_mutual_information(clean_fgco2, clean_index)
                    results['mutual_info'][i, j] = mi
                    results['mutual_info_sig'][i, j] = mi > 0.1  # Threshold for significance
                    
            except Exception:
                # Keep NaN values if calculation fails
                continue
    
    return results


def process_climate_index_multi(fgco2_data, index_data, index_name, periods, seasons, 
                                lat_dims, lon_dims):
    """
    Process multiple metrics for a single climate index across all periods.
    
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
        Dictionary containing all metric results for all periods
    """
    print(f"\nProcessing {index_name} multi-metric analysis...")
    
    # Initialize results dictionary
    results = {
        'pearson_corr': [],
        'pearson_pval': [],
        'pearson_sig': [],
        'spearman_corr': [],
        'spearman_pval': [],
        'spearman_sig': [],
        'mutual_info': [],
        'mutual_info_sig': []
    }
    
    # Store statistics for report
    stats_data = []
    
    for period in periods:
        print(f"  Calculating {period} metrics...")
        
        # Calculate period means
        fgco2_period = calculate_period_mean(fgco2_data, period, seasons)
        index_period = calculate_period_mean(index_data, period, seasons)
        
        # Calculate all metrics
        period_results = calculate_multi_metrics(fgco2_period, index_period, 
                                                 lat_dims, lon_dims)
        
        # Store results
        for key in results:
            results[key].append(period_results[key])
        
        # Calculate and store statistics
        stats_dict = {
            'index': index_name,
            'period': period
        }
        
        # Pearson statistics
        pearson_valid = ~np.isnan(period_results['pearson_corr'])
        if np.any(pearson_valid):
            stats_dict.update({
                'pearson_mean': np.nanmean(period_results['pearson_corr']),
                'pearson_std': np.nanstd(period_results['pearson_corr']),
                'pearson_min': np.nanmin(period_results['pearson_corr']),
                'pearson_max': np.nanmax(period_results['pearson_corr']),
                'pearson_sig_pct': 100 * np.sum(period_results['pearson_sig']) / np.sum(pearson_valid)
            })
        
        # Spearman statistics
        spearman_valid = ~np.isnan(period_results['spearman_corr'])
        if np.any(spearman_valid):
            stats_dict.update({
                'spearman_mean': np.nanmean(period_results['spearman_corr']),
                'spearman_std': np.nanstd(period_results['spearman_corr']),
                'spearman_min': np.nanmin(period_results['spearman_corr']),
                'spearman_max': np.nanmax(period_results['spearman_corr']),
                'spearman_sig_pct': 100 * np.sum(period_results['spearman_sig']) / np.sum(spearman_valid)
            })
        
        # Mutual Information statistics
        mi_valid = ~np.isnan(period_results['mutual_info'])
        if np.any(mi_valid):
            stats_dict.update({
                'mi_mean': np.nanmean(period_results['mutual_info']),
                'mi_std': np.nanstd(period_results['mutual_info']),
                'mi_min': np.nanmin(period_results['mutual_info']),
                'mi_max': np.nanmax(period_results['mutual_info']),
                'mi_sig_pct': 100 * np.sum(period_results['mutual_info_sig']) / np.sum(mi_valid)
            })
        
        stats_data.append(stats_dict)
        
        # Print summary
        print(f"    Pearson: mean={stats_dict.get('pearson_mean', np.nan):.3f}, "
              f"sig={stats_dict.get('pearson_sig_pct', 0):.1f}%")
        print(f"    Spearman: mean={stats_dict.get('spearman_mean', np.nan):.3f}, "
              f"sig={stats_dict.get('spearman_sig_pct', 0):.1f}%")
        print(f"    MI: mean={stats_dict.get('mi_mean', np.nan):.3f}, "
              f"sig={stats_dict.get('mi_sig_pct', 0):.1f}%")
    
    # Stack results
    for key in results:
        results[key] = np.stack(results[key])
    
    return results, stats_data


def create_multi_metric_dataset(dmi_results, oni_results, periods, lat_dims, lon_dims, 
                                time_ranges, creator_info):
    """
    Create the output NetCDF dataset with all metrics.
    
    Parameters
    ----------
    dmi_results : dict
        DMI multi-metric results
    oni_results : dict
        ONI multi-metric results
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
        # DMI Pearson variables
        'dmi_pearson_corr': xr.DataArray(
            dmi_results['pearson_corr'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Pearson correlation coefficient between fgCO2 and DMI',
                'standard_name': 'pearson_correlation_coefficient',
                'units': '1',
                'valid_range': [-1.0, 1.0]
            }
        ),
        'dmi_pearson_pval': xr.DataArray(
            dmi_results['pearson_pval'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Two-tailed p-value for DMI Pearson correlation test',
                'units': '1',
                'valid_range': [0.0, 1.0]
            }
        ),
        'dmi_pearson_sig': xr.DataArray(
            dmi_results['pearson_sig'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'DMI Pearson correlation significance at 95% confidence level',
                'units': '1',
                'flag_values': [0, 1],
                'flag_meanings': 'not_significant significant'
            }
        ),
        # DMI Spearman variables
        'dmi_spearman_corr': xr.DataArray(
            dmi_results['spearman_corr'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Spearman rank correlation coefficient between fgCO2 and DMI',
                'standard_name': 'spearman_correlation_coefficient',
                'units': '1',
                'valid_range': [-1.0, 1.0]
            }
        ),
        'dmi_spearman_pval': xr.DataArray(
            dmi_results['spearman_pval'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Two-tailed p-value for DMI Spearman correlation test',
                'units': '1',
                'valid_range': [0.0, 1.0]
            }
        ),
        'dmi_spearman_sig': xr.DataArray(
            dmi_results['spearman_sig'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'DMI Spearman correlation significance at 95% confidence level',
                'units': '1',
                'flag_values': [0, 1],
                'flag_meanings': 'not_significant significant'
            }
        ),
        # DMI Mutual Information variables
        'dmi_mutual_info': xr.DataArray(
            dmi_results['mutual_info'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Normalized mutual information between fgCO2 and DMI',
                'standard_name': 'mutual_information',
                'units': '1',
                'valid_range': [0.0, 1.0],
                'description': 'Normalized mutual information score (0=independent, 1=perfect dependence)'
            }
        ),
        'dmi_mutual_info_sig': xr.DataArray(
            dmi_results['mutual_info_sig'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'DMI mutual information significance (MI > 0.1)',
                'units': '1',
                'flag_values': [0, 1],
                'flag_meanings': 'not_significant significant'
            }
        ),
        # ONI Pearson variables
        'oni_pearson_corr': xr.DataArray(
            oni_results['pearson_corr'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Pearson correlation coefficient between fgCO2 and ONI',
                'standard_name': 'pearson_correlation_coefficient',
                'units': '1',
                'valid_range': [-1.0, 1.0]
            }
        ),
        'oni_pearson_pval': xr.DataArray(
            oni_results['pearson_pval'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Two-tailed p-value for ONI Pearson correlation test',
                'units': '1',
                'valid_range': [0.0, 1.0]
            }
        ),
        'oni_pearson_sig': xr.DataArray(
            oni_results['pearson_sig'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'ONI Pearson correlation significance at 95% confidence level',
                'units': '1',
                'flag_values': [0, 1],
                'flag_meanings': 'not_significant significant'
            }
        ),
        # ONI Spearman variables
        'oni_spearman_corr': xr.DataArray(
            oni_results['spearman_corr'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Spearman rank correlation coefficient between fgCO2 and ONI',
                'standard_name': 'spearman_correlation_coefficient',
                'units': '1',
                'valid_range': [-1.0, 1.0]
            }
        ),
        'oni_spearman_pval': xr.DataArray(
            oni_results['spearman_pval'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Two-tailed p-value for ONI Spearman correlation test',
                'units': '1',
                'valid_range': [0.0, 1.0]
            }
        ),
        'oni_spearman_sig': xr.DataArray(
            oni_results['spearman_sig'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'ONI Spearman correlation significance at 95% confidence level',
                'units': '1',
                'flag_values': [0, 1],
                'flag_meanings': 'not_significant significant'
            }
        ),
        # ONI Mutual Information variables
        'oni_mutual_info': xr.DataArray(
            oni_results['mutual_info'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'Normalized mutual information between fgCO2 and ONI',
                'standard_name': 'mutual_information',
                'units': '1',
                'valid_range': [0.0, 1.0],
                'description': 'Normalized mutual information score (0=independent, 1=perfect dependence)'
            }
        ),
        'oni_mutual_info_sig': xr.DataArray(
            oni_results['mutual_info_sig'],
            dims=['period', 'latitude', 'longitude'],
            coords={'period': period_coord, 'latitude': lat_dims, 'longitude': lon_dims},
            attrs={
                'long_name': 'ONI mutual information significance (MI > 0.1)',
                'units': '1',
                'flag_values': [0, 1],
                'flag_meanings': 'not_significant significant'
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
        'title': 'fgCO2 Climate Index Multi-Metric Analysis',
        'institution': 'University of California, Riverside',
        'source': 'Multi-metric analysis between fgCO2 and climate indices (DMI, ONI)',
        'metrics': 'Pearson correlation, Spearman correlation, Mutual Information',
        'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'references': 'Pearson and Spearman correlations with two-tailed significance tests; '
                     'Mutual Information using k-nearest neighbors estimation',
        'creator_name': creator_info['name'],
        'creator_email': creator_info['email'],
        'Conventions': 'CF-1.8',
        'project': 'fgCO2 Climate Teleconnections Multi-Metric Analysis',
        'processing_level': 'Level 3 - Derived geophysical variables',
        'period_definitions': ('Annual: Calendar year average; '
                             'DJF: December-January-February average; '
                             'MAM: March-April-May average; '
                             'JJA: June-July-August average; '
                             'SON: September-October-November average'),
        'statistical_significance': 'Correlations: p < 0.05 (95% confidence); MI: value > 0.1',
        'dmi_time_range': f"{pd.Timestamp(time_ranges['DMI'][0]).strftime('%Y-%m-%d')} to "
                          f"{pd.Timestamp(time_ranges['DMI'][1]).strftime('%Y-%m-%d')}",
        'oni_time_range': f"{pd.Timestamp(time_ranges['ONI'][0]).strftime('%Y-%m-%d')} to "
                          f"{pd.Timestamp(time_ranges['ONI'][1]).strftime('%Y-%m-%d')}"
    })
    
    return ds


def write_statistics_report(stats_data, output_file):
    """
    Write detailed statistics report to text file.
    
    Parameters
    ----------
    stats_data : list
        List of dictionaries containing statistics
    output_file : str
        Path to output text file
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("fgCO2 CLIMATE INDEX MULTI-METRIC ANALYSIS - STATISTICS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by index
        for index_name in ['DMI', 'ONI']:
            f.write(f"\n{index_name} INDEX ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            index_stats = [s for s in stats_data if s['index'] == index_name]
            
            for stat in index_stats:
                f.write(f"\nPeriod: {stat['period']}\n")
                f.write("~" * 20 + "\n")
                
                # Pearson correlation
                if 'pearson_mean' in stat:
                    f.write("\nPearson Correlation:\n")
                    f.write(f"  Mean:        {stat['pearson_mean']:7.4f}\n")
                    f.write(f"  Std Dev:     {stat['pearson_std']:7.4f}\n")
                    f.write(f"  Range:       [{stat['pearson_min']:7.4f}, {stat['pearson_max']:7.4f}]\n")
                    f.write(f"  Significant: {stat['pearson_sig_pct']:6.2f}%\n")
                
                # Spearman correlation
                if 'spearman_mean' in stat:
                    f.write("\nSpearman Correlation:\n")
                    f.write(f"  Mean:        {stat['spearman_mean']:7.4f}\n")
                    f.write(f"  Std Dev:     {stat['spearman_std']:7.4f}\n")
                    f.write(f"  Range:       [{stat['spearman_min']:7.4f}, {stat['spearman_max']:7.4f}]\n")
                    f.write(f"  Significant: {stat['spearman_sig_pct']:6.2f}%\n")
                
                # Mutual Information
                if 'mi_mean' in stat:
                    f.write("\nMutual Information:\n")
                    f.write(f"  Mean:        {stat['mi_mean']:7.4f}\n")
                    f.write(f"  Std Dev:     {stat['mi_std']:7.4f}\n")
                    f.write(f"  Range:       [{stat['mi_min']:7.4f}, {stat['mi_max']:7.4f}]\n")
                    f.write(f"  Significant: {stat['mi_sig_pct']:6.2f}% (MI > 0.1)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("NOTES:\n")
        f.write("- Pearson correlation: Linear relationship measure\n")
        f.write("- Spearman correlation: Monotonic relationship measure\n")
        f.write("- Mutual Information: General dependency measure (0=independent, 1=perfect)\n")
        f.write("- Significance thresholds: p<0.05 for correlations, MI>0.1 for mutual information\n")
        f.write("=" * 80 + "\n")


def main():
    """
    Main execution function for fgCO2 climate index multi-metric analysis.
    """
    print("=" * 80)
    print("fgCO2 Climate Index Multi-Metric Analysis")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    config = {
        'paths': {
            'fgco2': "../raw_data/fgco2_1998_2024.nc",
            'dmi': "../raw_data/dmi.nc",
            'oni': "../raw_data/oni.nc",
            'output': "../processed_data/fgco2_climate_indices_multimetric.nc",
            'stats': "../stats/multimetric_statistics.txt"
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
    
    # Ensure output directories exist
    ensure_directory(config['paths']['output'])
    ensure_directory(config['paths']['stats'])
    
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
    print(f"  Spatial grid: {len(lat_dims)} x {len(lon_dims)} (latitude x longitude)")
    
    # Dictionary to store time ranges and statistics
    time_ranges = {}
    all_stats = []
    
    # Process DMI multi-metric analysis
    print("\nDMI Multi-Metric Analysis:")
    print("-" * 40)
    fgco2_dmi, dmi_aligned, time_ranges['DMI'] = align_time_series(
        fgco2_data, dmi_data, "fgCO2", "DMI"
    )
    
    dmi_results, dmi_stats = process_climate_index_multi(
        fgco2_dmi, dmi_aligned, 'DMI', 
        config['periods'], config['seasons'], 
        lat_dims, lon_dims
    )
    all_stats.extend(dmi_stats)
    
    # Process ONI multi-metric analysis
    print("\nONI Multi-Metric Analysis:")
    print("-" * 40)
    fgco2_oni, oni_aligned, time_ranges['ONI'] = align_time_series(
        fgco2_data, oni_data, "fgCO2", "ONI"
    )
    
    oni_results, oni_stats = process_climate_index_multi(
        fgco2_oni, oni_aligned, 'ONI', 
        config['periods'], config['seasons'], 
        lat_dims, lon_dims
    )
    all_stats.extend(oni_stats)
    
    # Create output dataset
    print("\nCreating output dataset...")
    output_ds = create_multi_metric_dataset(
        dmi_results, oni_results, 
        config['periods'], lat_dims, lon_dims,
        time_ranges, config['creator']
    )
    
    # Save to NetCDF
    print(f"\nSaving results to: {config['paths']['output']}")
    encoding = {var: {'zlib': True, 'complevel': 4} for var in output_ds.data_vars}
    output_ds.to_netcdf(config['paths']['output'], encoding=encoding)
    
    # Write statistics report
    print(f"\nWriting statistics report to: {config['paths']['stats']}")
    write_statistics_report(all_stats, config['paths']['stats'])
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Multi-metric analysis completed successfully!")
    print(f"Output file: {config['paths']['output']}")
    print(f"Statistics report: {config['paths']['stats']}")
    print(f"File size: {os.path.getsize(config['paths']['output']) / 1024 / 1024:.1f} MB")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
