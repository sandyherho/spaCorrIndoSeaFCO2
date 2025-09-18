#!/usr/bin/env python
"""
Publication-Ready Multi-Metric Maps Visualization
==================================================

This script creates publication-quality figures of fgCO2 relationships
with climate indices (ONI and DMI) using three metrics (Pearson, Spearman, 
Mutual Information) for annual and seasonal periods.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: June 30, 2025
License: WTFPL

Output:
-------
Twelve figures saved in EPS, PNG, and PDF formats:
- For each metric (Pearson, Spearman, MI):
  1. Annual fgCO2-ONI metric
  2. Seasonal fgCO2-ONI metrics (2x2 layout)
  3. Annual fgCO2-DMI metric
  4. Seasonal fgCO2-DMI metrics (2x2 layout)
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
})


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def get_metric_config(metric_name):
    """
    Get configuration for specific metric.
    
    Parameters
    ----------
    metric_name : str
        Name of metric ('pearson', 'spearman', or 'mutual_info')
    
    Returns
    -------
    dict
        Configuration dictionary for the metric
    """
    configs = {
        'pearson': {
            'title': 'Pearson Correlation',
            'label': 'Pearson Correlation Coefficient',
            'vmin': -0.8,
            'vmax': 0.8,
            'cmap': 'RdBu_r',
            'extend': 'both',
            'ticks': np.arange(-0.8, 0.9, 0.2)
        },
        'spearman': {
            'title': 'Spearman Correlation',
            'label': 'Spearman Correlation Coefficient',
            'vmin': -0.8,
            'vmax': 0.8,
            'cmap': 'RdBu_r',
            'extend': 'both',
            'ticks': np.arange(-0.8, 0.9, 0.2)
        },
        'mutual_info': {
            'title': 'Mutual Information',
            'label': 'Normalized Mutual Information',
            'vmin': 0,
            'vmax': 0.5,
            'cmap': 'YlOrRd',
            'extend': 'max',
            'ticks': np.arange(0, 0.6, 0.1)
        }
    }
    return configs.get(metric_name, configs['pearson'])


def add_stippling_hatching(ax, lon, lat, significance, hatch_style='...', alpha=0.3):
    """
    Add stippling using hatching.
    
    Parameters
    ----------
    ax : matplotlib axes
        Axes to add stippling to
    lon : array
        Longitude coordinates
    lat : array
        Latitude coordinates
    significance : array
        Boolean array indicating significant points
    hatch_style : str
        Hatching pattern
    alpha : float
        Transparency of hatching
    """
    # Debug info
    n_sig = np.sum(significance)
    total = significance.size
    if total > 0:
        print(f"    Hatching: {n_sig} significant points out of {total} total ({100*n_sig/total:.1f}%)")
    
    if n_sig > 0:  # Only add hatching if there are significant points
        # Create significance contour
        sig_float = significance.astype(float)
        
        # Add hatching where significant (value = 1)
        cs = ax.contourf(lon, lat, sig_float, 
                         levels=[0.5, 1.5],  # Will capture values of 1
                         colors='none',
                         hatches=[hatch_style],
                         alpha=alpha,
                         transform=ccrs.PlateCarree(),
                         zorder=10)
        
        # Make the hatching more visible
        for collection in cs.collections:
            collection.set_edgecolor('black')
            collection.set_linewidth(0.5)


def create_metric_map(ax, data, lon, lat, significance=None, 
                     metric_config=None, add_features=True, 
                     gridlines=True, stipple_params=None):
    """
    Create a single metric map with stippling.
    
    Parameters
    ----------
    ax : matplotlib axes
        Axes with cartographic projection
    data : array
        2D metric data
    lon : array
        Longitude coordinates
    lat : array
        Latitude coordinates
    significance : array, optional
        Boolean array for stippling
    metric_config : dict
        Configuration for the metric
    add_features : bool
        Whether to add map features
    gridlines : bool
        Whether to add gridlines
    stipple_params : dict
        Parameters for stippling
    
    Returns
    -------
    mesh : pcolormesh object
        For colorbar creation
    """
    # Default stipple parameters
    if stipple_params is None:
        stipple_params = {'hatch_style': '..', 'alpha': 0.3}
    
    # Create the map
    mesh = ax.pcolormesh(lon, lat, data, 
                         transform=ccrs.PlateCarree(),
                         cmap=metric_config['cmap'], 
                         vmin=metric_config['vmin'], 
                         vmax=metric_config['vmax'],
                         rasterized=True, shading='nearest')
    
    # Add map features
    if add_features:
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5)
    
    # Add gridlines
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                         alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
    
    # Add stippling if significance data provided
    if significance is not None:
        sig_array = np.array(significance)
        if sig_array.shape == data.shape:
            add_stippling_hatching(ax, lon, lat, sig_array, **stipple_params)
        else:
            print(f"Warning: Significance shape {sig_array.shape} doesn't match data shape {data.shape}")
    
    # Set extent
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], 
                  crs=ccrs.PlateCarree())
    
    return mesh


def create_annual_figure(data, significance, lon, lat, index_name, metric_name, output_dir):
    """
    Create annual metric map figure.
    
    Parameters
    ----------
    data : xr.DataArray
        Annual metric data
    significance : xr.DataArray
        Annual significance data
    lon : array
        Longitude coordinates
    lat : array
        Latitude coordinates
    index_name : str
        Name of climate index (ONI or DMI)
    metric_name : str
        Name of metric (pearson, spearman, or mutual_info)
    output_dir : str
        Output directory path
    """
    # Get metric configuration
    metric_config = get_metric_config(metric_name)
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    
    # Create map
    mesh = create_metric_map(ax, data, lon, lat, 
                             significance=significance,
                             metric_config=metric_config)
    
    # Add title
    title = f'Annual fgCO₂-{index_name} {metric_config["title"]}'
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                        pad=0.05, fraction=0.046, 
                        extend=metric_config['extend'], shrink=0.8)
    cbar.set_label(metric_config['label'], fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(metric_config['ticks'])
    
    # Save in multiple formats
    base_filename = f'annual_fgco2_{index_name.lower()}_{metric_name}'
    for fmt in ['eps', 'png', 'pdf']:
        filepath = os.path.join(output_dir, f'{base_filename}.{fmt}')
        plt.savefig(filepath, format=fmt, bbox_inches='tight', 
                    pad_inches=0.1, dpi=300)
        print(f"  Saved: {filepath}")
    
    plt.close()


def create_seasonal_figure(data_dict, sig_dict, lon, lat, index_name, metric_name, output_dir):
    """
    Create 2x2 seasonal metric maps figure.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with seasonal metric data
    sig_dict : dict
        Dictionary with seasonal significance data
    lon : array
        Longitude coordinates
    lat : array
        Latitude coordinates
    index_name : str
        Name of climate index (ONI or DMI)
    metric_name : str
        Name of metric (pearson, spearman, or mutual_info)
    output_dir : str
        Output directory path
    """
    # Get metric configuration
    metric_config = get_metric_config(metric_name)
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    axes = axes.flatten()
    
    # Add main title
    main_title = f'Seasonal fgCO₂-{index_name} {metric_config["title"]}'
    fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.98)
    
    # Season order for 2x2 layout
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Create each seasonal map
    for i, season in enumerate(seasons):
        ax = axes[i]
        
        # Get data for this season
        metric_data = data_dict[season]
        sig_data = sig_dict[season]
        
        # Create map
        mesh = create_metric_map(ax, metric_data, lon, lat,
                                 significance=sig_data,
                                 metric_config=metric_config,
                                 gridlines=(i % 2 == 0))  # Only left panels
        
        # Add season label
        ax.text(0.02, 0.98, season, transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    # Add single colorbar for all subplots
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal',
                        extend=metric_config['extend'])
    cbar.set_label(metric_config['label'], fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(metric_config['ticks'])
    
    # Save in multiple formats
    base_filename = f'seasonal_fgco2_{index_name.lower()}_{metric_name}'
    for fmt in ['eps', 'png', 'pdf']:
        filepath = os.path.join(output_dir, f'{base_filename}.{fmt}')
        plt.savefig(filepath, format=fmt, bbox_inches='tight',
                    pad_inches=0.1, dpi=300)
        print(f"  Saved: {filepath}")
    
    plt.close()


def process_metric(ds, metric_name, index_name, lon, lat, output_dir):
    """
    Process and create figures for a specific metric.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    metric_name : str
        Name of metric (pearson, spearman, or mutual_info)
    index_name : str
        Name of climate index (ONI or DMI)
    lon : array
        Longitude coordinates
    lat : array
        Latitude coordinates
    output_dir : str
        Output directory path
    """
    index_lower = index_name.lower()
    
    # Variable names in the dataset
    if metric_name == 'mutual_info':
        data_var = f'{index_lower}_mutual_info'
        sig_var = f'{index_lower}_mutual_info_sig'
    else:
        data_var = f'{index_lower}_{metric_name}_corr'
        sig_var = f'{index_lower}_{metric_name}_sig'
    
    # Create annual figure
    print(f"  Creating annual {metric_name} map...")
    annual_data = ds[data_var].sel(period='Annual')
    annual_sig = ds[sig_var].sel(period='Annual')
    create_annual_figure(annual_data, annual_sig, lon, lat, 
                        index_name, metric_name, output_dir)
    
    # Create seasonal figure
    print(f"  Creating seasonal {metric_name} maps...")
    seasonal_data = {}
    seasonal_sig = {}
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        seasonal_data[season] = ds[data_var].sel(period=season)
        seasonal_sig[season] = ds[sig_var].sel(period=season)
    create_seasonal_figure(seasonal_data, seasonal_sig, lon, lat, 
                          index_name, metric_name, output_dir)


def main():
    """
    Main execution function for creating multi-metric map figures.
    """
    print("=" * 80)
    print("Creating Publication-Ready Multi-Metric Maps")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    input_file = "../processed_data/fgco2_climate_indices_multimetric.nc"
    output_dir = "../figs"
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Load data
    print("\nLoading multi-metric data...")
    try:
        ds = xr.open_dataset(input_file)
        print(f"  Loaded: {input_file}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract coordinates
    lon = ds.longitude.values
    lat = ds.latitude.values
    periods = ds.period.values
    
    print(f"  Grid dimensions: {len(lat)} x {len(lon)}")
    print(f"  Periods: {', '.join(periods)}")
    
    # Define metrics and indices to process
    metrics = ['pearson', 'spearman', 'mutual_info']
    indices = ['ONI', 'DMI']
    
    # Process each metric for each index
    for metric_name in metrics:
        metric_config = get_metric_config(metric_name)
        print(f"\n{'='*60}")
        print(f"Processing {metric_config['title']} Maps")
        print(f"{'='*60}")
        
        for index_name in indices:
            print(f"\n{index_name} {metric_config['title']} Figures:")
            print("-" * 40)
            process_metric(ds, metric_name, index_name, lon, lat, output_dir)
    
    # Close dataset
    ds.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("Multi-metric visualization completed successfully!")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    
    for metric_name in metrics:
        metric_label = get_metric_config(metric_name)['title']
        print(f"\n{metric_label}:")
        for index_name in ['oni', 'dmi']:
            print(f"  - annual_fgco2_{index_name}_{metric_name}.[eps/png/pdf]")
            print(f"  - seasonal_fgco2_{index_name}_{metric_name}.[eps/png/pdf]")
    
    print(f"\nTotal: {len(metrics) * len(indices) * 2} figure sets (36 files)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
