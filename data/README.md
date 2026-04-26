# Data downloaders

The pipeline expects daily, 0.5° NetCDF files in CF-conventions for each
configured variable. Three downloaders are provided:

| Script | Source | Use it for |
|---|---|---|
| `isimip_mpi_08_downloader.py` | ISIMIP3b bias-adjusted GCM data (MPI-ESM1-2-HR + H08 water-sector outputs) | Reproducing the manuscript projections (historical + SSP1-2.6 / 3-7.0 / 5-8.5) |
| `isimip_w5e5_downloader.py`   | ISIMIP3a W5E5 v2.0 reanalysis (the bias-adjustment reference for ISIMIP3b) | Sensitivity / observational baseline analyses |
| `isimip_era5_downloader.py`   | ISIMIP3a 20CRv3-ERA5 reanalysis (2018-2021) | Validating against Vaia (2018), W. Europe Floods (2021), Greece heat-drought (2021) |

All three respect the ISIMIP terms of use; please cite the underlying
datasets in your derived work.

## Quick examples

```bash
# Bias-adjustment reference — minimal 5-variable W5E5 pull, 1981-2010
python isimip_w5e5_downloader.py \
  --variables pr tasmax hurs sfcwind ps \
  --start-year 1981 --end-year 2010 \
  --out ./isimip_data/w5e5

# ISIMIP3b GCM data — uses an Excel file of UUIDs/links produced by the
# ISIMIP file-list export (see script header for the expected schema)
python isimip_mpi_08_downloader.py    # interactive / Colab-driven

# Validation reanalysis — 2018-2021 for the three known events
python isimip_era5_downloader.py
```

## Expected on-disk layout (matches the YAML configs)

```
isimip_data/
  isimip3b/
    pr_day_MPI-ESM1-2-HR_historical_*.nc
    pr_day_MPI-ESM1-2-HR_ssp585_*.nc
    ...
  w5e5/
    pr_W5E5v2.0_19810101-19901231.nc
    ...
  20crv3-era5/
    pr_*_2018-2021.nc
    ...
```

`Outputs/` and `*.nc` are git-ignored.
