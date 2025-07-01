# $F_{CO_{2}}$ Climate Teleconnections in Indonesian Seas

## Overview
This project analyzes the relationship between air-sea CO2 flux  and major climate indices (DMI and ONI) across the Indonesian maritime region. The Indonesian seas are a critical component of the global carbon cycle and are strongly influenced by both the Indian Ocean Dipole (IOD) and El Niño-Southern Oscillation (ENSO).


## Scripts

### 1. `calc.py` - Correlation Analysis
Calculates point-by-point Pearson correlations between $F_{CO_{2}}$ and climate indices.
```bash
python script/calc.py
```
- **Input**: fgCO2, DMI, and ONI NetCDF files (1998-2024)
- **Output**: Single NetCDF with correlations, p-values, and significance flags
- **Periods**: Annual, DJF, MAM, JJA, SON

### 2. `plot.py` - Publication-Ready Maps
Creates high-quality correlation maps with significance stippling.
```bash
python script/plot.py
```
- **Output**: 4 figures × 3 formats (EPS, PNG, PDF)
- **Features**: Hatching for p < 0.05, Pacific-centered projection

## Dependencies
```
xarray
numpy
scipy
pandas
matplotlib
cartopy
```

## Data Structure
```
raw_data/
├── fgco2_1998_2024.nc    # Air-sea CO2 flux
├── dmi.nc                 # Dipole Mode Index
└── oni.nc                 # Oceanic Niño Index

processed_data/
└── fgco2_climate_indices_correlations.nc

figs/
├── annual_fgco2_[oni|dmi]_correlation.[eps|png|pdf]
└── seasonal_fgco2_[oni|dmi]_correlation.[eps|png|pdf]
```



## Usage Example
```bash
# Run complete analysis
cd script/
python calc.py  # Calculate correlations
python plot.py  # Generate figures
```

## Output Interpretation
- **Red regions**: Positive correlation (increased DMI/ONI → increased CO2 outgassing)
- **Blue regions**: Negative correlation (increased DMI/ONI → increased CO2 uptake)
- **Stippling/Hatching**: Statistically significant correlations (p < 0.05)

## Author
Sandy Herho <sandy.herho@email.ucr.edu>  
University of California, Riverside

## License
WTFPL - Do What The F*** You Want To Public License

