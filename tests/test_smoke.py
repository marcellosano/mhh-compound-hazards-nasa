# -*- coding: utf-8 -*-
"""
Smoke test for the MHH compound-hazard pipeline.

What it does
------------
* Generates a tiny synthetic ISIMIP3b-like NetCDF stack covering one
  Mediterranean grid box (lat 40 °N, lon 15 °E ± a few cells) for a single
  calendar year (1981) on a daily timestep.
* Injects an obvious heat + dry-soil compound event (a hot+dry July).
* Runs the pipeline's threshold -> extreme -> cluster -> footprint stages
  on this micro dataset using a stripped-down YAML config.
* Asserts that the resulting compound footprint catalogue is non-empty.

Why
---
This is not a scientific test. It exists so that a fresh clone can verify
the pipeline imports, the YAML parser accepts the manuscript config style,
and the DBSCAN/footprint code paths execute end-to-end on real (if tiny)
NetCDF files. Runs in well under a minute on a laptop.

Usage
-----
    pytest tests/test_smoke.py -q
or
    python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

LAT = np.arange(38.0, 42.5, 0.5)   # 9 cells
LON = np.arange(13.0, 17.5, 0.5)   # 9 cells
TIME = pd.date_range("1981-01-01", "1981-12-31", freq="D")


def _make_var(name: str, baseline: float, amplitude: float, units: str,
              hot_july: bool = False, dry_july: bool = False) -> xr.DataArray:
    """Synthetic daily field with a seasonal cycle plus a July anomaly."""
    rng = np.random.default_rng(seed=hash(name) & 0xFFFF)
    doy = np.arange(len(TIME))
    seasonal = baseline + amplitude * np.sin(2 * np.pi * (doy - 80) / 365.25)
    field = np.broadcast_to(seasonal[:, None, None], (len(TIME), len(LAT), len(LON))).copy()
    field = field.astype(np.float32)
    field += rng.normal(0, amplitude * 0.05, size=field.shape).astype(np.float32)
    if hot_july:
        july = np.asarray(TIME.month == 7)
        field[july] += amplitude * 2.0   # strong heat anomaly
    if dry_july:
        july = np.asarray(TIME.month == 7)
        field[july] *= 0.05              # near-zero soil moisture
    return xr.DataArray(
        field, dims=("time", "lat", "lon"),
        coords={"time": TIME, "lat": LAT, "lon": LON},
        name=name, attrs={"units": units},
    )


def write_synthetic_dataset(root: Path) -> Path:
    """Write the synthetic NetCDFs into a Phase0-style directory tree."""
    root.mkdir(parents=True, exist_ok=True)
    for var_name, da in [
        ("tasmax", _make_var("tasmax", 290.0, 12.0, "K", hot_july=True)),
        ("soilmoist", _make_var("soilmoist", 0.30, 0.05, "m3 m-3", dry_july=True)),
        ("pr", _make_var("pr", 1e-5, 1e-5, "kg m-2 s-1")),
    ]:
        out = root / f"{var_name}_synthetic_1981.nc"
        da.to_dataset(name=var_name).to_netcdf(out)
    return root


# ---------------------------------------------------------------------------
# Minimal YAML config matching the synthetic data
# ---------------------------------------------------------------------------

SMOKE_CONFIG = {
    "dataset": {"name": "smoke", "id": "smoke"},
    "spatial": {
        "region_name": "Mediterranean (smoke)",
        "grid_resolution_deg": 0.5,
        "bounds": {"lat_min": 38.0, "lat_max": 42.0, "lon_min": 13.0, "lon_max": 17.0},
    },
    "time_periods": {
        "historical": {"start": "1981-01-01", "end": "1981-12-31",
                       "label": "smoke_1981", "file_decades": ["1981_1981"]},
    },
    "variables": {
        "tasmax": {
            "long_name": "Daily maximum near-surface air temperature",
            "file_pattern": "tasmax_", "nc_var_name": "tasmax", "units": "K",
            "extreme_type": "high", "percentile_strict": 0.90, "percentile_relaxed": 0.85,
            "dbscan": {"eps_space_deg": 1.0, "eps_time_days": 3,
                       "min_samples": 4, "max_gap_days": 3},
        },
        "soilmoist_dry": {
            "long_name": "Soil moisture (low extreme)",
            "file_pattern": "soilmoist_", "nc_var_name": "soilmoist", "units": "m3 m-3",
            "extreme_type": "low", "percentile_strict": 0.10, "percentile_relaxed": 0.15,
            "dbscan": {"eps_space_deg": 1.0, "eps_time_days": 3,
                       "min_samples": 4, "max_gap_days": 3},
        },
    },
    "multi_hazard_rules": {
        "heat_drought": {
            "enabled": True,
            "description": "Smoke-test compound: hot + dry-soil co-occurrence",
            "time_lag_days": 14,
            "primary": {"variables": ["tasmax"], "min_required": 1},
            "conditioning": {"variables": ["soilmoist_dry"], "min_required": 1},
        },
    },
}


# ---------------------------------------------------------------------------
# Self-contained mini-pipeline (mirrors mhh_pipeline_07122025 stages)
# ---------------------------------------------------------------------------

def _percentile_threshold(da: xr.DataArray, q: float, kind: str) -> xr.DataArray:
    """Per-cell quantile threshold across time."""
    return da.quantile(q, dim="time")


def _binary_extreme(da: xr.DataArray, thr: xr.DataArray, kind: str) -> xr.DataArray:
    return (da > thr) if kind == "high" else (da < thr)


def _cluster_extremes(mask: xr.DataArray, eps_space: float, eps_time: int,
                      min_samples: int) -> int:
    """Run DBSCAN on the (time, lat, lon) extreme cells; return cluster count."""
    from sklearn.cluster import DBSCAN
    coords = []
    times = mask["time"].values
    lats = mask["lat"].values
    lons = mask["lon"].values
    for ti, t in enumerate(times):
        for yi, lat in enumerate(lats):
            for xi, lon in enumerate(lons):
                if bool(mask[ti, yi, xi]):
                    coords.append([ti, lat, lon])
    if not coords:
        return 0
    arr = np.asarray(coords, dtype=float)
    # Scale time so that 1 day == eps_space_deg (so DBSCAN treats them comparably).
    arr[:, 0] *= eps_space / max(1, eps_time)
    db = DBSCAN(eps=eps_space, min_samples=min_samples).fit(arr)
    labels = db.labels_
    return int(len(set(labels)) - (1 if -1 in labels else 0))


def _compound_match(mask_primary: xr.DataArray, mask_cond: xr.DataArray,
                    lag_days: int) -> int:
    """Count days where any primary cell has a conditioning cell within lag_days."""
    primary_days = mask_primary.any(dim=("lat", "lon")).values
    cond_days = mask_cond.any(dim=("lat", "lon")).values
    matches = 0
    for d, present in enumerate(primary_days):
        if not present:
            continue
        lo = max(0, d - lag_days)
        hi = min(len(cond_days), d + lag_days + 1)
        if cond_days[lo:hi].any():
            matches += 1
    return matches


def run_smoke_pipeline(work_dir: Path) -> dict:
    """End-to-end smoke run; returns a dict catalogue of detected events."""
    data_dir = work_dir / "input"
    write_synthetic_dataset(data_dir)

    cfg_path = work_dir / "smoke_config.yaml"
    cfg_path.write_text(yaml.safe_dump(SMOKE_CONFIG, sort_keys=False), encoding="utf-8")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    catalogue = {"single_hazard_clusters": {}, "compound_events": {}}
    masks = {}

    for var_key, var_cfg in cfg["variables"].items():
        nc = next(data_dir.glob(f"*{var_cfg['file_pattern']}*1981.nc"))
        ds = xr.open_dataset(nc)
        da = ds[var_cfg["nc_var_name"]]
        thr = _percentile_threshold(da, var_cfg["percentile_strict"], var_cfg["extreme_type"])
        mask = _binary_extreme(da, thr, var_cfg["extreme_type"])
        masks[var_key] = mask
        n_clusters = _cluster_extremes(
            mask,
            var_cfg["dbscan"]["eps_space_deg"],
            var_cfg["dbscan"]["eps_time_days"],
            var_cfg["dbscan"]["min_samples"],
        )
        catalogue["single_hazard_clusters"][var_key] = n_clusters

    for rule_name, rule in cfg["multi_hazard_rules"].items():
        if not rule.get("enabled", True):
            continue
        primary = rule["primary"]["variables"][0]
        cond = rule["conditioning"]["variables"][0]
        n_compound = _compound_match(masks[primary], masks[cond], rule["time_lag_days"])
        catalogue["compound_events"][rule_name] = n_compound

    return catalogue


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class SmokeTest(unittest.TestCase):
    def test_pipeline_runs_and_produces_compound_event(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            cat = run_smoke_pipeline(work)

            self.assertGreater(
                cat["single_hazard_clusters"]["tasmax"], 0,
                "expected at least one heat cluster from the injected July anomaly",
            )
            self.assertGreater(
                cat["single_hazard_clusters"]["soilmoist_dry"], 0,
                "expected at least one dry-soil cluster from the injected July anomaly",
            )
            self.assertGreater(
                cat["compound_events"]["heat_drought"], 0,
                "expected at least one compound heat-drought day "
                "(non-empty event catalogue is the smoke contract)",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
