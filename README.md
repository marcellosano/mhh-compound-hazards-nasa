# MHH Compound Hazards Pipeline

A dataset-agnostic pipeline for detecting compound meteorological-hydrological hazards from gridded climate data. Identifies spatiotemporally coherent extreme events using percentile-based thresholding and DBSCAN clustering, then detects compound (multi-hazard) footprints via spatial and temporal co-occurrence rules.

## What it does

1. **Threshold calculation** -- Computes grid-cell-level percentile thresholds from a historical baseline period
2. **Extreme detection** -- Flags grid-cell-days exceeding thresholds as binary extremes
3. **DBSCAN clustering** -- Groups spatiotemporally contiguous extreme cells into coherent events, with gap-splitting for long-duration clusters
4. **Multi-hazard footprints** -- Matches co-occurring single-hazard events (e.g., extreme heat + drought) within configurable spatial distance and temporal lag windows

## Quick start (Google Colab)

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Run the pipeline script -- it loads config and launches the dashboard
# Set the config path before running:
import os
os.environ['MHH_CONFIG'] = '/content/drive/MyDrive/path/to/configs/isimip3b_europe.yaml'

# 3. Execute the pipeline file (all cells)
# The dashboard will appear with parameter controls and a "Save & Run Smart" button
```

## Configuration

The pipeline is fully configured via YAML files. See `configs/` for examples.

### 3-step workflow

1. **Define your variables** -- List the climate variables you want to analyze
2. **Set thresholds and clustering parameters** -- For each variable, configure percentile thresholds and DBSCAN spatial/temporal scales
3. **Define compound hazard rules** -- Specify which variable combinations constitute compound events

### Variable configuration reference

Each variable entry in the YAML `variables:` section supports these fields:

| Field | Required | Description |
|-------|----------|-------------|
| `long_name` | yes | Human-readable name |
| `file_pattern` | yes | String to match in filenames (e.g., `"_pr_"` or `"pr_day"`) |
| `nc_var_name` | yes | Variable name inside the NetCDF file |
| `units` | yes | Physical units (for documentation) |
| `extreme_type` | yes | `"high"` (detect exceedances) or `"low"` (detect deficits) |
| `percentile_strict` | yes | Percentile for threshold (0.0-1.0). E.g., 0.99 = 99th percentile |
| `percentile_relaxed` | yes | Relaxed percentile (used for sensitivity analysis) |
| `fixed_threshold` | no | Absolute threshold value. Combined with percentile via `use_fixed_as_minimum` |
| `use_fixed_as_minimum` | no | If true, extreme = (exceeds percentile AND exceeds fixed threshold) |
| `rolling_window_days` | no | Apply rolling sum before thresholding (e.g., 30 for precipitation deficit) |
| `derived` | no | If true, this variable is computed from other variables |
| `source_variables` | no | List of source variable names for derivation |
| `derivation` | no | Derivation function name (see below) |
| `dbscan.eps_space_deg` | yes | DBSCAN spatial neighbourhood radius (degrees) |
| `dbscan.eps_time_days` | yes | DBSCAN temporal neighbourhood radius (days) |
| `dbscan.min_samples` | yes | DBSCAN minimum cluster size |
| `dbscan.max_gap_days` | yes | Maximum temporal gap before splitting a cluster |

### Multi-hazard rules reference

Each rule in the `multi_hazard_rules:` section defines a compound hazard type:

```yaml
multi_hazard_rules:
  my_compound_hazard:
    enabled: true                    # set false to skip
    description: "Human-readable description"
    time_lag_days: 7                 # max temporal offset between primary and conditioning events
    primary:
      variables: ["var_key_1"]       # must reference keys from variables: section
      min_required: 1                # how many primary variables must have an event
    conditioning:
      variables: ["var_key_2", "var_key_3"]
      min_required: 1                # how many conditioning variables must co-occur
```

The pipeline finds primary single-hazard events, then searches for conditioning events within `time_lag_days` and a 500 km spatial window. Rules referencing variables not defined in the `variables:` section are automatically skipped with a warning.

### Derived variables

Some datasets provide raw component fields rather than the physical quantity the pipeline needs. For example, NEX-GDDP-CMIP6 provides eastward (`uas`) and northward (`vas`) wind components instead of scalar wind speed (`sfcwind`), and specific humidity (`huss`, in kg/kg) instead of relative humidity (`hurs`, in %). The pipeline can compute the required variables automatically at load time using built-in derivation functions — no preprocessing needed.

To use a derived variable, set `derived: true` in its config and specify the source variables and derivation function:

```yaml
sfcwind:
  derived: true
  source_variables: ["uas", "vas"]
  derivation: "windspeed_from_uv"
  # ... rest of config as normal ...
```

Built-in derivation functions:
- `windspeed_from_uv` -- Computes scalar wind speed from u/v components: `sqrt(uas^2 + vas^2)`. Use when your dataset provides separate eastward and northward wind fields.
- `rh_from_specific_humidity` -- Converts specific humidity (kg/kg) to relative humidity (%) using the Tetens formula for saturation vapour pressure, with temperature from `tasmax`. Use when your dataset provides `huss` instead of `hurs`.

If your dataset already provides a variable directly (e.g., ISIMIP3b provides `sfcwind` and `hurs` as-is), no derivation is needed — just point `file_pattern` and `nc_var_name` at the files.

### Example configs

- `configs/isimip3b_europe.yaml` -- ISIMIP3b bias-adjusted projections over Europe (0.5deg, 10 variables, 3 compound types)
- `configs/nexgddp_example.yaml` -- NASA NEX-GDDP-CMIP6 template (0.25deg, example variables with derivations)

## Requirements

```
pip install -r requirements.txt
```

Core dependencies: numpy, pandas, xarray, netcdf4, scikit-learn, matplotlib, seaborn, ipywidgets, pyyaml

## Citation

TBC

## License

TBC
