# MHH Compound Hazards Pipeline

A dataset-agnostic pipeline for detecting compound meteorological-hydrological
hazards from gridded climate data. Identifies spatiotemporally coherent
extreme events using percentile-based thresholding and DBSCAN clustering,
then detects compound (multi-hazard) footprints via spatial and temporal
co-occurrence rules.

This repository accompanies a manuscript in preparation for *Nature Climate
Change* (Sano et al.). It reproduces the late-century European analysis of
compound windstorm, flood, and heat-drought events under SSP1-2.6, SSP3-7.0,
and SSP5-8.5 forcing using ISIMIP3b bias-adjusted MPI-ESM1-2-HR projections,
and includes a NEX-GDDP-CMIP6 example configuration.

---

## Repository layout

```
.
├── README.md
├── LICENSE                                 (TBD — see Citation)
├── environment.yml                         conda env (recommended)
├── requirements.txt                        pip alternative
├── .gitignore                              excludes Outputs/ and *.nc
├── mhh_pipeline_07122025.py                core pipeline (6 classes)
├── configs/
│   ├── isimip3b_europe.yaml                manuscript config
│   └── nexgddp_example.yaml                NASA NEX-GDDP-CMIP6 template
├── data/
│   ├── README.md                           download guide
│   ├── isimip_mpi_08_downloader.py         ISIMIP3b GCM + H08 water-sector
│   ├── isimip_w5e5_downloader.py           W5E5 reanalysis (bias-adj. ref.)
│   └── isimip_era5_downloader.py           20CRv3-ERA5 (event validation)
├── analysis/
│   ├── mhh_analysis_light_23022026.py      late-century corrected analysis
│   └── build_publication_figures_26042026.py
└── tests/
    └── test_smoke.py                       end-to-end smoke test
```

Pipeline outputs (multi-GB NetCDF stacks under `Outputs/Runs/`) are
**git-ignored** and must be regenerated locally. See
[Reproduction](#reproduction) below.

---

## Quick start

```bash
git clone https://github.com/marcellosano/mhh-compound-hazards-nasa.git
cd mhh-compound-hazards-nasa

# Option A — conda (recommended, includes geopandas for figure base maps)
conda env create -f environment.yml
conda activate mhh

# Option B — pip
pip install -r requirements.txt

# Verify the pipeline runs end-to-end on synthetic data (~1 minute)
python tests/test_smoke.py
```

Expected output:

```
test_pipeline_runs_and_produces_compound_event ... ok
Ran 1 test in 53.386s
OK
```

---

## Reproduction

The published numbers can be reproduced in four steps.

### 1. Download the input data

Three download paths cover the manuscript:

```bash
# Bias-adjustment reference (W5E5 v2.0, 1981-2010, ~5 variables)
python data/isimip_w5e5_downloader.py \
  --variables pr tasmax hurs sfcwind ps \
  --start-year 1981 --end-year 2010 \
  --out ./isimip_data/w5e5

# ISIMIP3b GCM data (MPI-ESM1-2-HR + H08 water; historical + 3 SSPs)
# See script header for the expected Excel/UUID input.
python data/isimip_mpi_08_downloader.py

# Validation reanalysis (20CRv3-ERA5, 2018-2021, three known events)
python data/isimip_era5_downloader.py
```

See [`data/README.md`](data/README.md) for source citations and disk-space
guidance. The full ISIMIP3b pull is ~50–100 GB.

### 2. Run the pipeline

```bash
export MHH_CONFIG=./configs/isimip3b_europe.yaml
python mhh_pipeline_07122025.py
```

The pipeline writes a date-stamped run folder under
`Outputs/Runs/<run_name>/` containing:

- `Phase2_Thresholds/` — per-cell percentile thresholds
- `Phase2_Extremes/` — daily binary extreme masks
- `Phase3_Clusters/` — DBSCAN single-hazard cluster catalogues
- `Phase4_Footprints/` — `multihazard_<windstorm|flood|heat_drought_fire>.csv`
- `Phase5_Analysis/` — derived statistics
- `config.json`, `parameters_summary.txt` — exact parameters used

### 3. Reproduce the manuscript tables and headline numbers

```bash
python analysis/mhh_analysis_light_23022026.py
```

Writes `table1_frequency.csv` … `table4b_return_periods.csv` and
`key_findings.md` into `Outputs/Publication_Outputs/`.

This script applies the late-century correction (30 yr historical vs
30 yr 2071–2100), which fixes the 60 yr normalisation bug present in
earlier drafts.

### 4. Regenerate the publication figures

```bash
python analysis/build_publication_figures_26042026.py
```

Writes vector PDF + PNG for:

| File | Content |
|---|---|
| `fig1_pipeline.{pdf,png}` | Methodology schematic (5-stage flow) |
| `fig2_frequency.{pdf,png}` | Compound event frequency by hazard and SSP |
| `fig3_extremes.{pdf,png}` | Duration intensification + extreme event ratio |
| `fig4_regional_map.{pdf,png}` | AR6 regional change factors (heat-drought) |
| `fig5_return_periods.{pdf,png}` | Return-period changes for 3 validated events |
| `fig6_validation_map.{pdf,png}` | Detected events with future projection signal |

---

## Configuration

The pipeline is fully configured via YAML files. See `configs/` for examples.

### 3-step workflow

1. **Define your variables** — list the climate variables to analyse
2. **Set thresholds and clustering parameters** — for each variable, configure
   percentile thresholds and DBSCAN spatial/temporal scales
3. **Define compound hazard rules** — specify which variable combinations
   constitute compound events

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

The pipeline finds primary single-hazard events, then searches for
conditioning events within `time_lag_days` and a 500 km spatial window.
Rules referencing variables not defined in `variables:` are skipped with
a warning.

### Derived variables

Some datasets provide raw component fields rather than the physical quantity
the pipeline needs. For example, NEX-GDDP-CMIP6 provides eastward (`uas`)
and northward (`vas`) wind components instead of scalar wind speed
(`sfcwind`), and specific humidity (`huss`, in kg/kg) instead of relative
humidity (`hurs`, in %). The pipeline can compute the required variables
automatically at load time using built-in derivation functions — no
preprocessing needed.

```yaml
sfcwind:
  derived: true
  source_variables: ["uas", "vas"]
  derivation: "windspeed_from_uv"
  # ... rest of config as normal ...
```

Built-in derivation functions:

- `windspeed_from_uv` — Scalar wind speed from u/v components:
  `sqrt(uas^2 + vas^2)`. Use when your dataset provides separate eastward
  and northward wind fields.
- `rh_from_specific_humidity` — Specific humidity (kg/kg) → relative
  humidity (%) via Tetens saturation vapour pressure, with temperature from
  `tasmax`. Use when your dataset provides `huss` instead of `hurs`.

If your dataset already provides a variable directly (e.g., ISIMIP3b
provides `sfcwind` and `hurs` as-is), no derivation is needed — just
point `file_pattern` and `nc_var_name` at the files.

### Example configs

- `configs/isimip3b_europe.yaml` — ISIMIP3b bias-adjusted projections over
  Europe (0.5°, 10 variables, 3 compound types)
- `configs/nexgddp_example.yaml` — NASA NEX-GDDP-CMIP6 template (0.25°,
  example variables with derivations)

---

## Smoke test

`tests/test_smoke.py` runs the full pipeline on a single Mediterranean grid
box for one calendar year of synthetic data. It injects a hot + dry-soil
July anomaly and asserts that the resulting compound footprint catalogue
is non-empty. Use it to confirm a fresh clone is healthy:

```bash
python tests/test_smoke.py        # standalone
pytest tests/test_smoke.py -q     # via pytest if installed
```

Expect ~1 minute runtime on a laptop. The test does **not** require
downloaded ISIMIP data.

---

## Citation

If you use this pipeline, please cite:

> Sano, M., Ferrario, D. M., Torresan, S., Critto, A. (in preparation).
> Compound meteorological-hydrological hazard detection using DBSCAN
> clustering of climate projections. *Nature Climate Change*.

A Zenodo archive of this repository will be linked here at the time of
manuscript submission for the Code Availability statement.

## License

License is to be determined and will be added at the time of publication.
Until then, the contents of this repository are made available for
academic review and reproduction of the accompanying manuscript.
