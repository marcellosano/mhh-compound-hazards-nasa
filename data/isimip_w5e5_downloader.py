# -*- coding: utf-8 -*-
"""
isimip_w5e5_downloader.py
=========================
Minimal downloader for the ISIMIP W5E5 v2.0 observational reanalysis dataset
used as the bias-adjustment baseline for the ISIMIP3b GCM track.

W5E5 is daily, 0.5°, global, distributed by ISIMIP. This script uses the
official isimip-client package to discover files matching a variable list
and a year range, and downloads them with resume support.

Reference:
    Lange, S., Menz, C., Gleixner, S. et al. (2021). WFDE5 over land
    merged with ERA5 over the ocean (W5E5 v2.0). ISIMIP Repository.
    https://doi.org/10.48364/ISIMIP.342217

Usage (CLI):
    python isimip_w5e5_downloader.py \
        --variables pr tasmax hurs sfcwind ps \
        --start-year 1981 --end-year 2010 \
        --out ./isimip_data/w5e5

Notes:
    - Some variables (qtot, soilmoist) are not available in W5E5 directly;
      they come from H08 (or another impact model) forced with W5E5.
      Use isimip_mpi_08_downloader.py for those, pointing at the
      `secondary/water_global/h08/w5e5v2.0` tree.
    - Files are large (~50 MB / variable / decade). Plan disk accordingly.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from isimip_client.client import ISIMIPClient
except ImportError:
    print("ERROR: isimip-client is required. Install with:")
    print("    pip install isimip-client")
    sys.exit(1)


W5E5_PATH = "ISIMIP3a/InputData/climate/atmosphere/obsclim/global/daily/historical/W5E5v2.0"


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("w5e5_downloader")
    log.setLevel(logging.INFO)
    if not log.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        log.addHandler(sh)
        fh = logging.FileHandler(out_dir / "w5e5_download.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    return log


def discover_files(variables, start_year, end_year, log):
    """Return a list of file URLs from the ISIMIP repository."""
    client = ISIMIPClient()
    log.info(f"Querying ISIMIP repository for W5E5 variables {variables}")
    response = client.datasets(
        simulation_round="ISIMIP3a",
        product="InputData",
        category="climate",
        climate_forcing="w5e5v2.0",
        climate_scenario="obsclim",
        time_step="daily",
        climate_variable=variables,
    )
    urls = []
    for ds in response["results"]:
        for f in ds.get("files", []):
            name = f.get("name", "")
            url = f.get("file_url")
            year_token = name.split("_")[-1].replace(".nc", "")
            try:
                file_start = int(year_token.split("-")[0][:4])
                file_end = int(year_token.split("-")[1][:4])
            except Exception:
                file_start = file_end = None
            if file_start is None or (file_end >= start_year and file_start <= end_year):
                urls.append((name, url))
    log.info(f"Discovered {len(urls)} matching files")
    return urls


def download_with_resume(url: str, dest: Path, log, chunk: int = 1 << 15) -> bool:
    """Stream-download `url` to `dest` with resume support."""
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=120)
        if r.status_code in (200, 206):
            total = int(r.headers.get("content-length", 0)) + existing
            mode = "ab" if existing else "wb"
            with open(dest, mode) as f, tqdm(
                total=total, initial=existing, unit="B", unit_scale=True, desc=dest.name
            ) as pbar:
                for block in r.iter_content(chunk_size=chunk):
                    if block:
                        f.write(block)
                        pbar.update(len(block))
            return True
        log.error(f"HTTP {r.status_code} for {url}")
        return False
    except Exception as exc:
        log.error(f"Download failed for {url}: {exc}")
        return False


def main():
    p = argparse.ArgumentParser(description="Download ISIMIP3a W5E5 v2.0 reanalysis files.")
    p.add_argument("--variables", nargs="+", required=True,
                   help="Climate variables (e.g. pr tasmax hurs sfcwind ps)")
    p.add_argument("--start-year", type=int, default=1981)
    p.add_argument("--end-year", type=int, default=2010)
    p.add_argument("--out", type=Path, default=Path("./isimip_data/w5e5"))
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()

    out_dir = args.out
    log = setup_logging(out_dir)

    files = discover_files(args.variables, args.start_year, args.end_year, log)
    if not files:
        log.error("No files matched. Check variable names and year range.")
        sys.exit(2)

    fails = []
    for name, url in files:
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            log.info(f"Skipping {name} (already present)")
            continue
        for attempt in range(1, args.max_retries + 1):
            log.info(f"Downloading {name} (attempt {attempt}/{args.max_retries})")
            if download_with_resume(url, dest, log):
                break
            time.sleep(2 ** attempt)
        else:
            fails.append(name)

    if fails:
        log.error(f"Failed to download {len(fails)} files: {fails}")
        sys.exit(3)
    log.info("All files downloaded successfully.")


if __name__ == "__main__":
    main()
