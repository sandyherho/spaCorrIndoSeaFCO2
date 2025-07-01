#!/usr/bin/env python
"""
Publication-Ready Correlation Maps Visualization
================================================

This script creates publication-quality figures of fgCO2 correlations
with climate indices (ONI and DMI) for annual and seasonal periods.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: June 30, 2025
License: WTFPL

Output:
-------
Four figures saved in EPS, PNG, and PDF formats:
1. Annual fgCO2-ONI correlation
2. Seasonal fgCO2-ONI correlations (2x2 layout)
3. Annual fgCO2-DMI correlation
4. Seasonal fgCO2-DMI correlations (2x2 layout)
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
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


def add_stippling(ax, lon, lat, significance, density=3, size=3, alpha=0.8):
    """
    Add stippling to indicate statistical significance.
    
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
    density : int
        Density of stippling (lower = more dense)
    size : float
        Size of stipple dots
    alpha : float
        Transparency of stipples
    """
    # Create meshgrid
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    # Apply density reduction - but keep enough points for visible pattern
    sig_points = significance[::density, ::density]
    lon_points = lon_mesh[::density, ::density]
    lat_points = lat_mesh[::density, ::density]
    
    # Plot stipples where significant
    mask = sig_points.astype(bool)  # Ensure boolean type
    
    # Debug info
    n_sig = np.sum(mask)
    print(f"    Stippling: {n_sig} significant points out of {mask.size} total points")
    
    if np.any(mask):  # Only plot if there are significant points
        # Use larger, more visible dots
        ax.scatter(lon_points[mask], lat_points[mask], 
                   s=size, c='black', alpha=alpha, 
                   transform=ccrs.PlateCarree(), 
                   marker='o', edgecolors='none', 
                   zorder=10,  # Ensure stipples are on top
                   rasterized=True)  # Rasterize for smaller file size
        
def add_stippling_hatching(ax, lon, lat, significance, hatch_style='...', alpha=0.3):
    """
    Add stippling using hatching (alternative method, often clearer in publications).
    
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
        Hatching pattern ('...', '///', '\\\\\\', '|||', '+++', 'xxx')
    alpha : float
        Transparency of hatching
    """
    # Debug info
    n_sig = np.sum(significance)
    total = significance.size
    print(f"    Hatching: {n_sig} significant points out of {total} total ({100*n_sig/total:.1f}%)")
    
    if n_sig > 0:  # Only add hatching if there are significant points
        # Create significance contour
        # Convert boolean to float for contouring
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


def create_correlation_map(ax, data, lon, lat, significance=None, 
                          vmin=-0.8, vmax=0.8, cmap='RdBu_r',
                          add_features=True, gridlines=True,
                          stipple_params=None, stipple_method='dots'):
    """
    Create a single correlation map with stippling.
    
    Parameters
    ----------
    ax : matplotlib axes
        Axes with cartographic projection
    data : array
        2D correlation data
    lon : array
        Longitude coordinates
    lat : array
        Latitude coordinates
    significance : array, optional
        Boolean array for stippling
    vmin, vmax : float
        Colorbar limits
    cmap : str
        Colormap name
    add_features : bool
        Whether to add map features
    gridlines : bool
        Whether to add gridlines
    stipple_params : dict
        Parameters for stippling
    stipple_method : str
        'dots' for scatter points or 'hatching' for contour hatching
    
    Returns
    -------
    mesh : pcolormesh object
        For colorbar creation
    """
    # Default stipple parameters - increased size and visibility
    if stipple_params is None:
        if stipple_method == 'dots':
            stipple_params = {'density': 2, 'size': 8, 'alpha': 0.7}
        else:
            stipple_params = {'hatch_style': '...', 'alpha': 0.3}
    
    # Create the correlation map
    mesh = ax.pcolormesh(lon, lat, data, 
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, vmin=vmin, vmax=vmax,
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
        # Convert to numpy array and ensure correct shape
        sig_array = np.array(significance)
        if sig_array.shape == data.shape:
            if stipple_method == 'dots':
                add_stippling(ax, lon, lat, sig_array, **stipple_params)
            elif stipple_method == 'hatching':
                add_stippling_hatching(ax, lon, lat, sig_array, **stipple_params)
        else:
            print(f"Warning: Significance shape {sig_array.shape} doesn't match data shape {data.shape}")
    
    # Set extent (adjust based on your data coverage)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], 
                  crs=ccrs.PlateCarree())
    
    return mesh


def create_annual_figure(data, significance, lon, lat, index_name, output_dir):
    """
    Create annual correlation map figure.
    
    Parameters
    ----------
    data : xr.DataArray
        Annual correlation data
    significance : xr.DataArray
        Annual significance data
    lon : array
        Longitude coordinates
    lat : array
        Latitude coordinates
    index_name : str
        Name of climate index (ONI or DMI)
    output_dir : str
        Output directory path
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    
    # Create correlation map
    mesh = create_correlation_map(ax, data, lon, lat, 
                                  significance=significance,
                                  vmin=-0.8, vmax=0.8,
                                  stipple_method='hatching',
                                  stipple_params={'hatch_style': '...', 'alpha': 0.35})
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                        pad=0.05, fraction=0.046, 
                        extend='both', shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Set ticks
    cbar.set_ticks(np.arange(-0.8, 0.9, 0.2))
    
    # Save in multiple formats
    base_filename = f'annual_fgco2_{index_name.lower()}_correlation'
    for fmt in ['eps', 'png', 'pdf']:
        filepath = os.path.join(output_dir, f'{base_filename}.{fmt}')
        plt.savefig(filepath, format=fmt, bbox_inches='tight', 
                    pad_inches=0.1, dpi=300)
        print(f"Saved: {filepath}")
    
    plt.close()


def create_seasonal_figure(data_dict, sig_dict, lon, lat, index_name, output_dir):
    """
    Create 2x2 seasonal correlation maps figure.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with seasonal correlation data
    sig_dict : dict
        Dictionary with seasonal significance data
    lon : array
        Longitude coordinates
    lat : array
        Latitude coordinates
    index_name : str
        Name of climate index (ONI or DMI)
    output_dir : str
        Output directory path
    """
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    axes = axes.flatten()
    
    # Season order for 2x2 layout
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Create each seasonal map
    for i, season in enumerate(seasons):
        ax = axes[i]
        
        # Get data for this season
        corr_data = data_dict[season]
        sig_data = sig_dict[season]
        
        # Create map
        mesh = create_correlation_map(ax, corr_data, lon, lat,
                                      significance=sig_data,
                                      vmin=-0.8, vmax=0.8,
                                      gridlines=(i % 2 == 0),  # Only left panels
                                      stipple_method='hatching',
                                      stipple_params={'hatch_style': '..', 'alpha': 0.3})
        
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
                        extend='both')
    cbar.set_label('Correlation Coefficient', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(np.arange(-0.8, 0.9, 0.2))
    
    # Save in multiple formats
    base_filename = f'seasonal_fgco2_{index_name.lower()}_correlation'
    for fmt in ['eps', 'png', 'pdf']:
        filepath = os.path.join(output_dir, f'{base_filename}.{fmt}')
        plt.savefig(filepath, format=fmt, bbox_inches='tight',
                    pad_inches=0.1, dpi=300)
        print(f"Saved: {filepath}")
    
    plt.close()


def main():
    """
    Main execution function for creating correlation map figures.
    """
    print("=" * 80)
    print("Creating Publication-Ready Correlation Maps")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    input_file = "../processed_data/fgco2_climate_indices_correlations.nc"
    output_dir = "../figs"
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Load data
    print("\nLoading correlation data...")
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
    
    # Process ONI correlations
    print("\nCreating ONI correlation figures...")
    
    # Figure 1: Annual fgCO2-ONI correlation
    print("  Creating annual ONI correlation map...")
    annual_oni_corr = ds.oni_correlation.sel(period='Annual')
    annual_oni_sig = ds.oni_significant.sel(period='Annual')
    create_annual_figure(annual_oni_corr, annual_oni_sig, lon, lat, 'ONI', output_dir)
    
    # Figure 2: Seasonal fgCO2-ONI correlations
    print("  Creating seasonal ONI correlation maps...")
    oni_seasonal_data = {}
    oni_seasonal_sig = {}
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        oni_seasonal_data[season] = ds.oni_correlation.sel(period=season)
        oni_seasonal_sig[season] = ds.oni_significant.sel(period=season)
    create_seasonal_figure(oni_seasonal_data, oni_seasonal_sig, lon, lat, 'ONI', output_dir)
    
    # Process DMI correlations
    print("\nCreating DMI correlation figures...")
    
    # Figure 3: Annual fgCO2-DMI correlation
    print("  Creating annual DMI correlation map...")
    annual_dmi_corr = ds.dmi_correlation.sel(period='Annual')
    annual_dmi_sig = ds.dmi_significant.sel(period='Annual')
    create_annual_figure(annual_dmi_corr, annual_dmi_sig, lon, lat, 'DMI', output_dir)
    
    # Figure 4: Seasonal fgCO2-DMI correlations
    print("  Creating seasonal DMI correlation maps...")
    dmi_seasonal_data = {}
    dmi_seasonal_sig = {}
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        dmi_seasonal_data[season] = ds.dmi_correlation.sel(period=season)
        dmi_seasonal_sig[season] = ds.dmi_significant.sel(period=season)
    create_seasonal_figure(dmi_seasonal_data, dmi_seasonal_sig, lon, lat, 'DMI', output_dir)
    
    # Close dataset
    ds.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("Visualization completed successfully!")
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print("  - annual_fgco2_oni_correlation.[eps/png/pdf]")
    print("  - seasonal_fgco2_oni_correlation.[eps/png/pdf]")
    print("  - annual_fgco2_dmi_correlation.[eps/png/pdf]")
    print("  - seasonal_fgco2_dmi_correlation.[eps/png/pdf]")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()