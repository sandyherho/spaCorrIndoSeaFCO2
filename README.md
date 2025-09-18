# Multi-Metric Analysis of Air-Sea CO₂ Flux Teleconnections in Indonesian Maritime Region

## Abstract

This repository contains a comprehensive analysis framework for investigating relationships between air-sea CO₂ flux ($F_{CO_2}$) and major climate indices across the Indonesian maritime region. We employ three complementary statistical metrics—Pearson correlation, Spearman rank correlation, and Mutual Information—to capture linear, monotonic, and nonlinear dependencies respectively. The Indonesian seas represent a critical component of the global carbon cycle and are strongly influenced by both the Indian Ocean Dipole (IOD) and El Niño-Southern Oscillation (ENSO) phenomena.

## Table of Contents

- [Mathematical Framework](#mathematical-framework)
  - [Pearson Correlation](#pearson-correlation-coefficient)
  - [Spearman Correlation](#spearman-rank-correlation)
  - [Mutual Information](#mutual-information)
- [Data Processing](#data-processing)
- [Installation](#installation)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Interpretation Guide](#interpretation-guide)
- [Citation](#citation)

## Mathematical Framework

### Pearson Correlation Coefficient

The Pearson correlation coefficient $\rho_{X,Y}$ measures the linear relationship between $F_{CO_2}$ and climate indices. For time series $\mathbf{X} = \{x_i\}_{i=1}^n$ (fgCO₂) and $\mathbf{Y} = \{y_i\}_{i=1}^n$ (climate index), the Pearson correlation is defined as:

$$\rho_{X,Y} = \frac{\sigma_{XY}}{\sigma_X \sigma_Y} = \frac{\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]}{\sqrt{\mathbb{E}[(X - \mu_X)^2]}\sqrt{\mathbb{E}[(Y - \mu_Y)^2]}}$$

The sample estimate $r$ is given by:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

where:
- $\mu_X$, $\mu_Y$ are population means; $\bar{x}$, $\bar{y}$ are sample means
- $\sigma_X$, $\sigma_Y$ are population standard deviations
- $\sigma_{XY}$ is the covariance
- $\mathbb{E}[\cdot]$ denotes the expectation operator

**Statistical Significance:** We compute the test statistic:

$$\tau = r\sqrt{\frac{n-2}{1-r^2}} \sim t_{n-2}$$

where $\tau$ follows Student's t-distribution with $\nu = n - 2$ degrees of freedom. The null hypothesis $H_0: \rho = 0$ is rejected when $p < \alpha = 0.05$.

### Spearman Rank Correlation

The Spearman correlation $\rho_s$ captures monotonic relationships by operating on ranked data. Let $\mathcal{R}(x_i)$ and $\mathcal{R}(y_i)$ denote the ranks of observations:

$$\rho_s = 1 - \frac{6\sum_{i=1}^{n}\delta_i^2}{n(n^2-1)}$$

where $\delta_i = \mathcal{R}(x_i) - \mathcal{R}(y_i)$ is the rank difference.

Equivalently, using the Pearson correlation on ranks:

$$\rho_s = \frac{\sum_{i=1}^{n}(\xi_i - \bar{\xi})(\psi_i - \bar{\psi})}{\sqrt{\sum_{i=1}^{n}(\xi_i - \bar{\xi})^2}\sqrt{\sum_{i=1}^{n}(\psi_i - \bar{\psi})^2}}$$

where $\xi_i = \mathcal{R}(x_i)$ and $\psi_i = \mathcal{R}(y_i)$.

**Properties:** 
- Invariant under monotonic transformations
- Robust to outliers: $\|\nabla \rho_s\|_\infty < \|\nabla \rho\|_\infty$
- Distribution-free under $H_0$

### Mutual Information

Mutual Information $\mathcal{I}(X;Y)$ quantifies the reduction in uncertainty about one variable given knowledge of another:

$$\mathcal{I}(X;Y) = \mathcal{H}(X) - \mathcal{H}(X|Y) = \mathcal{H}(Y) - \mathcal{H}(Y|X)$$

where $\mathcal{H}(\cdot)$ denotes Shannon entropy. For continuous variables with joint density $f_{X,Y}(x,y)$ and marginal densities $f_X(x)$, $f_Y(y)$:

$$\mathcal{I}(X;Y) = \iint_{\Omega} f_{X,Y}(x,y) \log\left(\frac{f_{X,Y}(x,y)}{f_X(x)f_Y(y)}\right) dx dy$$

Alternatively, using Kullback-Leibler divergence:

$$\mathcal{I}(X;Y) = D_{KL}(f_{X,Y} \| f_X \otimes f_Y)$$

**k-NN Estimation:** Following Kraskov-Stögbauer-Grassberger algorithm with $k=3$ neighbors:

$$\hat{\mathcal{I}}(X;Y) = \psi(n) - \langle\psi(n_x^{(i)} + 1) + \psi(n_y^{(i)} + 1)\rangle_i + \psi(k)$$

where:
- $\psi(\cdot)$ is the digamma function: $\psi(z) = \frac{d}{dz}\log\Gamma(z)$
- $n_x^{(i)}$, $n_y^{(i)}$ are marginal neighbor counts
- $\langle \cdot \rangle_i$ denotes sample average

**Normalization:** To obtain $\mathcal{I}_{norm} \in [0,1]$:

$$\mathcal{I}_{norm} = \min\left(\frac{\hat{\mathcal{I}}(X;Y)}{\min(\mathcal{H}(X), \mathcal{H}(Y))}, 1\right) \approx \min\left(\frac{\hat{\mathcal{I}}(X;Y)}{\log n}, 1\right)$$

**Significance Criterion:** $\mathcal{I}_{norm} > \theta = 0.1$ indicates significant dependency.

## Data Processing

### Temporal Aggregation

For each period $\mathcal{P} \in \{\text{Annual}, \text{DJF}, \text{MAM}, \text{JJA}, \text{SON}\}$, the temporal mean is computed as:

$$\bar{X}_\mathcal{P} = \frac{1}{|\mathcal{T}_\mathcal{P}|}\sum_{t \in \mathcal{T}_\mathcal{P}} X_t$$

where $\mathcal{T}_\mathcal{P} = \{t : t \in \mathcal{P}\}$ represents the temporal subset.

### Seasonal Definitions

Let $\Phi$ denote the set of months, then:
- **DJF:** $\mathcal{T}_{DJF} = \{t : \phi(t) \in \{12, 1, 2\}\}$ (Austral summer)
- **MAM:** $\mathcal{T}_{MAM} = \{t : \phi(t) \in \{3, 4, 5\}\}$ (Austral autumn)
- **JJA:** $\mathcal{T}_{JJA} = \{t : \phi(t) \in \{6, 7, 8\}\}$ (Austral winter)
- **SON:** $\mathcal{T}_{SON} = \{t : \phi(t) \in \{9, 10, 11\}\}$ (Austral spring)

where $\phi(t)$ returns the month of timestamp $t$.

### Grid-Point Analysis

For each spatial coordinate $(\lambda_i, \varphi_j)$ where $\lambda$ denotes longitude and $\varphi$ denotes latitude:

$$\mathcal{M}_{\alpha}(\lambda_i, \varphi_j) = f_{\alpha}(\mathbf{F}_{CO_2}(\lambda_i, \varphi_j, \cdot), \mathbf{I}(\cdot))$$

where:
- $\mathcal{M}_{\alpha} \in \{\rho, \rho_s, \mathcal{I}_{norm}\}$ represents the metric
- $\mathbf{F}_{CO_2}(\lambda_i, \varphi_j, \cdot)$ is the time series at grid point $(\lambda_i, \varphi_j)$
- $\mathbf{I}(\cdot)$ is the climate index time series
- $f_{\alpha}$ is the corresponding metric function

## Installation

### Prerequisites

```bash
# Create conda environment
conda create -n fgco2_analysis python=3.9
conda activate fgco2_analysis

# Install dependencies
conda install -c conda-forge xarray numpy scipy pandas matplotlib cartopy scikit-learn
pip install netCDF4
```

### Clone Repository

```bash
git clone https://github.com/sandyherho/spaCorrIndoSeaFCO2
cd spaCorrIndoSeaFCO2
```

## Usage

### Complete Analysis Pipeline

```bash
# Step 1: Calculate all metrics and generate statistics
cd script/
python calc.py

# Step 2: Generate publication-ready visualizations
python plot.py
```

### Individual Script Usage

#### 1. Multi-Metric Calculation (`calc.py`)

Computes three metrics $\{\rho, \rho_s, \mathcal{I}\}$ for all grid points and periods:

```python
#!/usr/bin/env python
# Basic usage
python script/calc.py

# Output files:
# - ../processed_data/fgco2_climate_indices_multimetric.nc
# - ../stats/multimetric_statistics.txt
```

**Key Functions:**
- `calculate_multi_metrics()`: Computes $\rho$, $\rho_s$, and $\mathcal{I}$ simultaneously
- `calculate_mutual_information()`: k-NN based $\mathcal{I}$ estimation
- `write_statistics_report()`: Generates comprehensive statistics

#### 2. Visualization (`plot.py`)

Creates publication-quality maps for each metric:

```python
#!/usr/bin/env python
# Generate all figures
python script/plot.py

# Creates 36 files (3 metrics × 2 indices × 2 temporal × 3 formats)
```

**Customization Options:**

```python
# Modify metric configurations in get_metric_config()
configs = {
    'pearson': {
        'vmin': -0.8, 'vmax': 0.8,  # Adjust color scale
        'cmap': 'RdBu_r',           # Change colormap
        'extend': 'both'             # Colorbar extension
    },
    'mutual_info': {
        'vmin': 0, 'vmax': 0.5,
        'cmap': 'YlOrRd',
        'extend': 'max'
    }
}
```

### Advanced Usage

#### Process Specific Time Periods

```python
import xarray as xr
from calc import process_climate_index_multi

# Load your data
fgco2 = xr.open_dataset('path/to/fgco2.nc')
index = xr.open_dataset('path/to/index.nc')

# Process only specific periods
periods = ['Annual', 'JJA']  # Summer only
results, stats = process_climate_index_multi(
    fgco2['fgco2'], index['value'], 
    'CustomIndex', periods, seasons, 
    lat_dims, lon_dims
)
```

#### Extract Regional Statistics

```python
# Load results
ds = xr.open_dataset('../processed_data/fgco2_climate_indices_multimetric.nc')

# Extract region (e.g., Java Sea)
java_sea = ds.sel(
    latitude=slice(-8, -4),
    longitude=slice(106, 114)
)

# Compute regional mean correlation
regional_pearson = java_sea.oni_pearson_corr.mean(['latitude', 'longitude'])
print(f"Java Sea mean Pearson ρ: {regional_pearson.values}")
```

## Output Structure

### Directory Organization

```
project/
├── raw_data/
│   ├── fgco2_1998_2024.nc        # Air-sea CO₂ flux data
│   ├── dmi.nc                     # Dipole Mode Index
│   └── oni.nc                     # Oceanic Niño Index
│
├── processed_data/
│   └── fgco2_climate_indices_multimetric.nc  # All metrics
│
├── stats/
│   └── multimetric_statistics.txt            # Statistical summary
│
├── figs/
│   ├── annual_fgco2_oni_pearson.[eps|png|pdf]
│   ├── annual_fgco2_oni_spearman.[eps|png|pdf]
│   ├── annual_fgco2_oni_mutual_info.[eps|png|pdf]
│   ├── seasonal_fgco2_oni_pearson.[eps|png|pdf]
│   ├── seasonal_fgco2_oni_spearman.[eps|png|pdf]
│   ├── seasonal_fgco2_oni_mutual_info.[eps|png|pdf]
│   └── ... (same structure for DMI)
│
└── script/
    ├── calc.py                    # Multi-metric calculation
    └── plot.py                    # Visualization
```

### NetCDF Variables Structure

The output NetCDF contains 16 variables per climate index:

```python
# For each index (dmi, oni) and metric (pearson, spearman, mutual_info):
{index}_{metric}_corr      # Metric values [period, lat, lon]
{index}_{metric}_pval      # P-values (correlations only)
{index}_{metric}_sig       # Significance flags

# Example variable access:
import xarray as xr
ds = xr.open_dataset('fgco2_climate_indices_multimetric.nc')
pearson_oni = ds.oni_pearson_corr.sel(period='Annual')
significance = ds.oni_pearson_sig.sel(period='Annual')
```

## Interpretation Guide

### Metric Comparison

| Metric | Symbol | Range | Interpretation | Optimal Use Case |
|--------|--------|-------|----------------|------------------|
| **Pearson** | $\rho$ | $[-1, 1]$ | Linear relationship strength | Simple linear dependencies |
| **Spearman** | $\rho_s$ | $[-1, 1]$ | Monotonic relationship strength | Ranked relationships, outlier-robust |
| **Mutual Info** | $\mathcal{I}$ | $[0, 1]$ | General dependency ($0$=independent) | Complex nonlinear patterns |

### Color Scheme Interpretation

#### Correlation Maps ($\rho$ and $\rho_s$)
- **Deep Blue** ($\rho < -0.6$): Strong negative correlation
  - $\uparrow$ Climate Index $\rightarrow$ $\downarrow$ CO₂ outgassing (ocean uptake)
- **Light Blue** ($-0.6 \leq \rho < -0.2$): Weak negative correlation
- **White** ($-0.2 \leq \rho \leq 0.2$): No significant correlation
- **Light Red** ($0.2 < \rho \leq 0.6$): Weak positive correlation
- **Deep Red** ($\rho > 0.6$): Strong positive correlation
  - $\uparrow$ Climate Index $\rightarrow$ $\uparrow$ CO₂ outgassing

#### Mutual Information Maps ($\mathcal{I}_{norm}$)
- **Yellow** ($0 \leq \mathcal{I} < 0.1$): Weak/no dependency
- **Orange** ($0.1 \leq \mathcal{I} < 0.3$): Moderate dependency
- **Red** ($\mathcal{I} \geq 0.3$): Strong dependency

### Significance Indicators

**Hatching Pattern (..)**: Indicates statistically significant relationships
- Correlations: $p < \alpha = 0.05$ (95% confidence level)
- Mutual Information: $\mathcal{I}_{norm} > \theta = 0.1$ threshold

### Physical Interpretation

**Positive DMI (Indian Ocean Dipole):**
- Warmer western Indian Ocean ($\Delta T > 0$)
- Enhanced upwelling in eastern Indonesian seas
- Increased $F_{CO_2}$ (outgassing)

**Positive ONI (El Niño):**
- Suppressed upwelling in eastern Pacific
- Modified Indonesian Throughflow ($\Psi_{ITF}$)
- Complex regional $F_{CO_2}$ responses

## Statistical Summary Example

```
DMI INDEX - Annual Period
----------------------------------------
Pearson Correlation (ρ):
  Mean μ:      0.1654 ± 0.2931
  Range:      [-0.6582, 0.7388]
  Significant: 33.83% of grid points

Spearman Correlation (ρₛ):
  Mean μ:      0.1738 ± 0.2675
  Range:      [-0.6274, 0.6752]
  Significant: 27.10% of grid points

Mutual Information (𝒢):
  Mean μ:      0.0323 ± 0.0343
  Range:      [0.0000, 0.1664]
  Significant: 5.66% of grid points
```

## Performance Considerations

- **Memory Usage:** $\mathcal{O}(n \times m \times p)$ where $n$=lat, $m$=lon, $p$=time
- **Computational Complexity:** 
  - Pearson/Spearman: $\mathcal{O}(n \times m \times p)$
  - Mutual Information: $\mathcal{O}(n \times m \times p \times k \log k)$ for k-NN
- **Processing Time:** $\sim$10-15 minutes for $1° \times 1°$ global grid
- **Parallelization:** Consider using `dask` for larger datasets:

```python
import dask.array as da
# Convert to dask arrays for parallel processing
fgco2_dask = da.from_array(fgco2_data, chunks={'time': 100})
```

## Mathematical Properties

### Correlation Bounds

For any two random variables $X$ and $Y$:

$$|\rho_{X,Y}| \leq 1 \text{ and } |\rho_s| \leq 1$$

with equality if and only if there exists a perfect linear (for $\rho$) or monotonic (for $\rho_s$) relationship.

### Information-Theoretic Relationships

The mutual information relates to correlation through:

$$\mathcal{I}(X;Y) \geq -\frac{1}{2}\log(1 - \rho^2_{X,Y})$$

for jointly Gaussian variables, with equality when $(X,Y) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{herho2025fgco2,
  author = {Herho, Sandy H. S.},
  title = {Multi-Metric Analysis of Air-Sea CO₂ Flux Teleconnections},
  year = {2025},
  institution = {University of California, Riverside},
  email = {sandy.herho@email.ucr.edu},
  license = {WTFPL}
}
```

## Author

**Sandy H. S. Herho**  
Department of Earth and Planetary Sciences  
University of California, Riverside  
Email: sandy.herho@email.ucr.edu

## License

WTFPL - Do What The F*** You Want To Public License

See [LICENSE](LICENSE) for details.

---

*Last Updated: September 2025*
