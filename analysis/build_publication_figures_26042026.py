# -*- coding: utf-8 -*-
"""
build_publication_figures_26042026.py
=====================================
Regenerates the NCC publication figure set with corrected terminology.

Changes vs mhh_analysis_light_23022026.py:
    - Compound Fire Weather  ->  Compound Heat-Drought  (display label only;
      internal hazard key 'heat_drought_fire' is preserved so the upstream
      pipeline outputs are unchanged).
    - Output filenames shifted by one to make room for Figure 1 (the pipeline
      schematic):
          fig1_frequency       -> fig2_frequency
          fig2_extremes        -> fig3_extremes
          fig3_regional_map    -> fig4_regional_map
          fig4_return_periods  -> fig5_return_periods
          fig5_validation_map  -> fig6_validation_map
    - New Figure 1: vector pipeline schematic modelled on slide 6 of
      AGU 2025 MH Marcello Sano_v4_17122025_upload.pptx
      (5 vertical stages: ISIMIP -> Thresholds -> Single-Hazard Extremes ->
       Single-hazard Clusters -> Multi-hazard Footprint).
    - Local Windows paths instead of Colab paths.

This script is the figure-renaming pass for v6 of the manuscript.
Upstream analysis logic is unchanged.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ---------------------------------------------------------------------------
# Paths and labels
# ---------------------------------------------------------------------------

PROJECT_ROOT = r"C:\Users\msano\OneDrive - unive.it\Publications\MS NCC MH"
RUN_PATH = os.path.join(PROJECT_ROOT, "Outputs", "Runs", "MHH_DBSCAN_Europe_20251202_v7")
VALIDATION_PATH = os.path.join(PROJECT_ROOT, "Outputs", "Validation_ERA5")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Outputs", "Publication_Outputs")

HISTORICAL_YEARS = 30
FUTURE_YEARS_PER_SSP = 30
LATE_CENTURY_START = 2071

SCENARIOS = ["historical", "ssp126", "ssp370", "ssp585"]
SCENARIO_LABELS = {
    "historical": "Historical\n(1981-2010)",
    "ssp126": "SSP1-2.6\n(2071-2100)",
    "ssp370": "SSP3-7.0\n(2071-2100)",
    "ssp585": "SSP5-8.5\n(2071-2100)",
}
COLORS = {
    "historical": "#404040",
    "ssp126": "#2166AC",
    "ssp370": "#F4A582",
    "ssp585": "#B2182B",
    "windstorm": "#7570B3",
    "flood": "#1B9E77",
    "heat_drought_fire": "#D95F02",
}
HAZARD_NAMES = {
    "windstorm": "Compound Windstorm",
    "flood": "Compound Flood",
    "heat_drought_fire": "Compound Heat-Drought",
}
AR6_REGIONS = {
    "NEU": {"name": "Northern Europe", "lat": 62.5, "lon": 15},
    "WCE": {"name": "W & C Europe", "lat": 50, "lon": 7.5},
    "EEU": {"name": "Eastern Europe", "lat": 50, "lon": 32.5},
    "MED": {"name": "Mediterranean", "lat": 40, "lon": 15},
}

FIG_DPI = 300
FIG_FORMATS = ["png", "pdf"]
FONT_SIZE = 10


def setup_style():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "sans-serif",
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": 100,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,   # editable text in PDF
        "ps.fonttype": 42,
    })


def save_figure(fig, name):
    for fmt in FIG_FORMATS:
        path = os.path.join(OUTPUT_PATH, f"{name}.{fmt}")
        fig.savefig(path, format=fmt, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  saved: {name}.png/pdf")


# ---------------------------------------------------------------------------
# Country boundaries (with offline fallback)
# ---------------------------------------------------------------------------

def _load_europe_boundaries():
    """Return a GeoDataFrame of European country boundaries, or None."""
    try:
        import geopandas as gpd
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        return world.cx[-15:45, 33:72]
    except Exception as e:
        print(f"  note: country boundaries unavailable ({e}); plotting plain background")
        return None


# ---------------------------------------------------------------------------
# Data loading (mirror of mhh_analysis_light_23022026.load_data)
# ---------------------------------------------------------------------------

def load_data():
    print("\nLoading footprint catalogues and validation data")
    data = {"footprints": {}, "validation": {}}

    fp_dir = os.path.join(RUN_PATH, "Phase6_Footprints")
    if not os.path.exists(fp_dir):
        fp_dir = os.path.join(RUN_PATH, "Phase4_Footprints")

    for hazard in HAZARD_NAMES.keys():
        filepath = os.path.join(fp_dir, f"multihazard_{hazard}.csv")
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found")
            continue
        df = pd.read_csv(filepath)
        df["start_date"] = pd.to_datetime(df["start_day"])
        df["year"] = df["start_date"].dt.year
        n_before = len(df)
        df = df[(df["scenario"] == "historical") | (df["year"] >= LATE_CENTURY_START)]
        n_filtered = n_before - len(df)
        data["footprints"][hazard] = df
        print(f"  {HAZARD_NAMES[hazard]}: {len(df):,} events (filtered {n_filtered:,} mid-century)")

    fp_file = os.path.join(VALIDATION_PATH, "event_fingerprints_v3.csv")
    if os.path.exists(fp_file):
        data["validation"]["fingerprints"] = pd.read_csv(fp_file)
        print(f"  validation fingerprints: {len(data['validation']['fingerprints'])} events")

    exc_file = os.path.join(VALIDATION_PATH, "future_exceedance_v3.csv")
    if os.path.exists(exc_file):
        data["validation"]["exceedance"] = pd.read_csv(exc_file)

    return data


# ---------------------------------------------------------------------------
# Analyses (verbatim from mhh_analysis_light_23022026, but using local labels)
# ---------------------------------------------------------------------------

def analyze_frequency(data):
    rows = []
    for hazard, df in data["footprints"].items():
        name = HAZARD_NAMES[hazard]
        counts = df.groupby("scenario").size()
        hist = counts.get("historical", 0)
        hist_rate = hist / HISTORICAL_YEARS * 10
        for scenario in ["ssp126", "ssp370", "ssp585"]:
            count = counts.get(scenario, 0)
            rate = count / FUTURE_YEARS_PER_SSP * 10
            change = rate / hist_rate if hist_rate > 0 else np.nan
            rows.append({
                "Hazard": name, "Hazard_Key": hazard, "Scenario": scenario,
                "Historical_Count": hist, "Historical_Rate": hist_rate,
                "Future_Count": count, "Future_Rate": rate,
                "Change_Factor": change,
            })
    return pd.DataFrame(rows)


def assign_region(lat, lon):
    if lat >= 55:
        return "NEU"
    elif lat >= 45:
        return "WCE" if lon < 25 else "EEU"
    return "MED"


def analyze_characteristics(data):
    rows = []
    for hazard, df in data["footprints"].items():
        name = HAZARD_NAMES[hazard]
        hist = df[df["scenario"] == "historical"]
        if len(hist) == 0:
            continue
        hist_dur_p95 = hist["duration_days"].quantile(0.95)
        hist_dur_p90 = hist["duration_days"].quantile(0.90)
        hist_ext_p95 = hist["spatial_extent_km2"].quantile(0.95)
        n_hist_extreme = len(hist[hist["duration_days"] >= hist_dur_p90])
        hist_extreme_rate = n_hist_extreme / HISTORICAL_YEARS * 10
        for scenario in ["ssp126", "ssp370", "ssp585"]:
            future = df[df["scenario"] == scenario]
            if len(future) == 0:
                continue
            fut_dur_p95 = future["duration_days"].quantile(0.95)
            fut_ext_p95 = future["spatial_extent_km2"].quantile(0.95)
            n_extreme = len(future[future["duration_days"] >= hist_dur_p90])
            extreme_rate = n_extreme / FUTURE_YEARS_PER_SSP * 10
            extreme_change = extreme_rate / hist_extreme_rate if hist_extreme_rate > 0 else np.nan
            rows.append({
                "Hazard": name, "Hazard_Key": hazard, "Scenario": scenario,
                "P95_Ratio": fut_dur_p95 / hist_dur_p95 if hist_dur_p95 > 0 else np.nan,
                "Extent_P95_Ratio": fut_ext_p95 / hist_ext_p95 if hist_ext_p95 > 0 else np.nan,
                "Extreme_Events_Change": extreme_change,
            })
    return pd.DataFrame(rows)


def analyze_regional(data):
    rows = []
    for hazard, df in data["footprints"].items():
        name = HAZARD_NAMES[hazard]
        df = df.copy()
        df["region"] = df.apply(lambda r: assign_region(r["center_lat"], r["center_lon"]), axis=1)
        for region_code, region_info in AR6_REGIONS.items():
            region_df = df[df["region"] == region_code]
            hist = len(region_df[region_df["scenario"] == "historical"])
            hist_rate = hist / HISTORICAL_YEARS * 10
            for scenario in ["ssp126", "ssp370", "ssp585"]:
                future = len(region_df[region_df["scenario"] == scenario])
                future_rate = future / FUTURE_YEARS_PER_SSP * 10
                change = future_rate / hist_rate if hist_rate > 0 else np.nan
                rows.append({
                    "Hazard": name, "Hazard_Key": hazard,
                    "Region": region_info["name"], "Region_Code": region_code,
                    "Scenario": scenario,
                    "Historical_Rate": hist_rate, "Future_Rate": future_rate,
                    "Change_Factor": change,
                })
    return pd.DataFrame(rows)


def analyze_validation(data):
    table4a = pd.DataFrame()
    return_rows = []
    if "fingerprints" not in data["validation"] or len(data["footprints"]) == 0:
        return table4a, pd.DataFrame()
    fp = data["validation"]["fingerprints"]

    val_rows = []
    for _, row in fp.iterrows():
        val_rows.append({
            "Event": row.get("event_name", "Unknown"),
            "Detection": "Footprint",
            "Duration_Days": row.get("duration_days", 0),
            "Extent_km2": row.get("spatial_extent_km2", 0),
            "Primary_Variable": row.get("primary_variable", ""),
            "Conditioning_Variables": row.get("conditioning_variables", ""),
        })
    table4a = pd.DataFrame(val_rows)

    event_configs = {
        "Vaia Windstorm": {"hazard": "windstorm", "min_extent_frac": 0.5},
        "Western Europe Floods": {"hazard": "flood", "min_extent_frac": 0.5},
        "Greece Fire Weather": {"hazard": "heat_drought_fire", "min_extent_frac": 0.5},
    }
    for _, row in fp.iterrows():
        event_name = str(row.get("event_name", "")).split(" - ")[0]
        if event_name in event_configs:
            dur = int(row.get("duration_days", 1))
            extent = float(row.get("spatial_extent_km2", 0))
            event_configs[event_name]["min_duration"] = max(1, dur - 2)
            event_configs[event_name]["min_extent"] = extent * event_configs[event_name]["min_extent_frac"]

    for event_name, cfg in event_configs.items():
        hazard = cfg["hazard"]
        if hazard not in data["footprints"]:
            continue
        df = data["footprints"][hazard]
        min_dur = cfg.get("min_duration", 1)
        min_ext = cfg.get("min_extent", 0)
        matching = df[(df["duration_days"] >= min_dur) & (df["spatial_extent_km2"] >= min_ext)]
        counts = {s: len(matching[matching["scenario"] == s]) for s in SCENARIOS}
        rates = {
            "historical": counts["historical"] / HISTORICAL_YEARS * 10,
            "ssp126": counts["ssp126"] / FUTURE_YEARS_PER_SSP * 10,
            "ssp370": counts["ssp370"] / FUTURE_YEARS_PER_SSP * 10,
            "ssp585": counts["ssp585"] / FUTURE_YEARS_PER_SSP * 10,
        }
        returns = {k: (10 / v if v > 0 else np.inf) for k, v in rates.items()}
        ssp585_change = returns["historical"] / returns["ssp585"] if returns["ssp585"] > 0 else np.nan
        return_rows.append({
            "Event": event_name,
            "Hist_Rate_Per_Decade": rates["historical"],
            "Hist_Return_Years": returns["historical"],
            "SSP126_Rate": rates["ssp126"], "SSP126_Return_Years": returns["ssp126"],
            "SSP370_Rate": rates["ssp370"], "SSP370_Return_Years": returns["ssp370"],
            "SSP585_Rate": rates["ssp585"], "SSP585_Return_Years": returns["ssp585"],
            "SSP585_Change_Factor": ssp585_change,
        })
    return table4a, pd.DataFrame(return_rows)


# ---------------------------------------------------------------------------
# Figure 1 -- Pipeline schematic (NEW)
# ---------------------------------------------------------------------------

def plot_pipeline_schematic():
    """5-stage vertical pipeline modelled on slide 6 of the AGU 2025 deck."""
    fig, ax = plt.subplots(figsize=(8.0, 9.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")

    box_w, box_h = 4.6, 1.25
    cx = 6.6  # boxes anchored to right column; left column for annotations
    y_centres = [9.4, 7.6, 5.8, 4.0, 2.2]

    stages = [
        {
            "title": "ISIMIP3b reanalysis & GCM projections",
            "subtitle": "MPI-ESM1-2-HR, 0.5° daily, 7 variables",
            "color": "#FBE5D6",
            "edge": "#E07B39",
            "annotation": (
                "Inputs\n"
                "Bias-adjusted daily fields:\n"
                "pr, tasmax, hurs, sfcWind, ps,\n"
                "qtot, soilmoist (1981–2100,\n"
                "historical + SSP1-2.6 / 3-7.0 / 5-8.5)"
            ),
        },
        {
            "title": "Empirical & statistical thresholds",
            "subtitle": "Per-cell percentiles + fixed criteria",
            "color": "#DBE9F4",
            "edge": "#2166AC",
            "annotation": (
                "Thresholds\n"
                "Grid-cell percentiles (1981–2010\n"
                "baseline) plus absolute floors;\n"
                "yields binary extreme masks."
            ),
        },
        {
            "title": "Single-hazard extremes",
            "subtitle": "Daily binary extreme masks per variable",
            "color": "#DBE9F4",
            "edge": "#2166AC",
            "annotation": (
                "Detection\n"
                "Each grid-cell-day flagged where\n"
                "the variable exceeds (or falls below)\n"
                "its calibrated threshold."
            ),
        },
        {
            "title": "Single-hazard clusters",
            "subtitle": "DBSCAN spatiotemporal grouping",
            "color": "#DBE9F4",
            "edge": "#2166AC",
            "annotation": (
                "Clustering\n"
                "DBSCAN groups contiguous extreme\n"
                "cells in space-time; gap-splitting\n"
                "preserves event identity."
            ),
        },
        {
            "title": "Multi-hazard footprints",
            "subtitle": "Compound co-occurrence (≤500 km, lag rules)",
            "color": "#DBE9F4",
            "edge": "#2166AC",
            "annotation": (
                "Compound footprints\n"
                "Primary + conditioning clusters\n"
                "matched within event-specific\n"
                "spatial and temporal windows:\n"
                "windstorm, flood, heat-drought."
            ),
        },
    ]

    # Boxes + titles
    for stage, yc in zip(stages, y_centres):
        x_left = cx - box_w / 2
        y_bot = yc - box_h / 2
        box = FancyBboxPatch(
            (x_left, y_bot), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.4, edgecolor=stage["edge"], facecolor=stage["color"],
        )
        ax.add_patch(box)
        ax.text(cx, yc + 0.22, stage["title"], ha="center", va="center",
                fontsize=11, fontweight="bold", color="#202020")
        ax.text(cx, yc - 0.28, stage["subtitle"], ha="center", va="center",
                fontsize=8.5, color="#404040", style="italic")

    # Down arrows between consecutive boxes
    for y_top, y_bot in zip(y_centres[:-1], y_centres[1:]):
        arrow = FancyArrowPatch(
            (cx, y_top - box_h / 2), (cx, y_bot + box_h / 2),
            arrowstyle="-|>", mutation_scale=18,
            linewidth=1.6, color="#404040",
        )
        ax.add_patch(arrow)

    # Left-column annotations
    annot_x = 0.25
    for stage, yc in zip(stages, y_centres):
        ax.annotate(
            stage["annotation"],
            xy=(cx - box_w / 2 - 0.15, yc), xycoords="data",
            xytext=(annot_x, yc), textcoords="data",
            fontsize=8.2, ha="left", va="center", color="#202020",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#B0B0B0", linewidth=0.6),
            arrowprops=dict(arrowstyle="-", color="#A0A0A0", lw=0.8,
                            connectionstyle="arc3,rad=0"),
        )

    ax.text(5.0, 10.6, "Pipeline overview",
            ha="center", va="center", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 -- Frequency bars (was fig1_frequency)
# ---------------------------------------------------------------------------

def plot_frequency(table1):
    fig, ax = plt.subplots(figsize=(7, 4))
    hazards = list(HAZARD_NAMES.keys())
    x = np.arange(len(hazards))
    width = 0.2
    for i, scenario in enumerate(["historical", "ssp126", "ssp370", "ssp585"]):
        if scenario == "historical":
            rates = [table1[(table1["Hazard_Key"] == h)]["Historical_Rate"].iloc[0] for h in hazards]
        else:
            rates = [table1[(table1["Hazard_Key"] == h) & (table1["Scenario"] == scenario)]["Future_Rate"].iloc[0]
                     for h in hazards]
        ax.bar(x + (i - 1.5) * width, rates, width,
               label=SCENARIO_LABELS[scenario], color=COLORS[scenario],
               edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Events per decade")
    ax.set_xticks(x)
    ax.set_xticklabels([HAZARD_NAMES[h] for h in hazards])
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    for i, h in enumerate(hazards):
        row = table1[(table1["Hazard_Key"] == h) & (table1["Scenario"] == "ssp585")].iloc[0]
        ax.annotate(f"{row['Change_Factor']:.2f}×",
                    xy=(i + 1.5 * width, row["Future_Rate"]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 -- Extremes (was fig2_extremes)
# ---------------------------------------------------------------------------

def plot_extremes(table2):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    hazards = list(HAZARD_NAMES.keys())
    x = np.arange(len(hazards))
    width = 0.25

    ax = axes[0]
    for i, scenario in enumerate(["ssp126", "ssp370", "ssp585"]):
        ratios = []
        for h in hazards:
            row = table2[(table2["Hazard_Key"] == h) & (table2["Scenario"] == scenario)]
            ratios.append(row["P95_Ratio"].iloc[0] if len(row) else 1.0)
        ax.bar(x + (i - 1) * width, ratios, width,
               label=SCENARIO_LABELS[scenario], color=COLORS[scenario],
               edgecolor="white", linewidth=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("P95 Duration Ratio\n(Future / Historical)")
    ax.set_xticks(x)
    ax.set_xticklabels([HAZARD_NAMES[h].replace(" ", "\n") for h in hazards], fontsize=8)
    ax.set_title("A. Duration Intensification", fontweight="bold", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    ax = axes[1]
    for i, scenario in enumerate(["ssp126", "ssp370", "ssp585"]):
        changes = []
        for h in hazards:
            row = table2[(table2["Hazard_Key"] == h) & (table2["Scenario"] == scenario)]
            changes.append(row["Extreme_Events_Change"].iloc[0] if len(row) else 1.0)
        ax.bar(x + (i - 1) * width, changes, width,
               label=SCENARIO_LABELS[scenario], color=COLORS[scenario],
               edgecolor="white", linewidth=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("Extreme Events Ratio\n(exceeding hist. P90)")
    ax.set_xticks(x)
    ax.set_xticklabels([HAZARD_NAMES[h].replace(" ", "\n") for h in hazards], fontsize=8)
    ax.set_title("B. Extreme Event Frequency", fontweight="bold", loc="left")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 -- Regional map (was fig3_regional_map)
# ---------------------------------------------------------------------------

def plot_regional(table3):
    fig, ax = plt.subplots(figsize=(9, 7))
    europe = _load_europe_boundaries()
    if europe is not None:
        europe.plot(ax=ax, color="#f0f0f0", edgecolor="gray", linewidth=0.5)
    else:
        ax.set_facecolor("#e6f2ff")

    ssp585 = table3[(table3["Scenario"] == "ssp585") & (table3["Hazard_Key"] == "heat_drought_fire")]
    for region_code, region_info in AR6_REGIONS.items():
        row = ssp585[ssp585["Region_Code"] == region_code]
        if len(row) == 0:
            continue
        change = row["Change_Factor"].iloc[0]
        if pd.isna(change):
            facecolor = "lightgray"
        elif change >= 2:
            facecolor = "#B2182B"
        elif change >= 1.5:
            facecolor = "#F4A582"
        elif change > 1:
            facecolor = "#FDDBC7"
        else:
            facecolor = "#92C5DE"
        circle = plt.Circle((region_info["lon"], region_info["lat"]), 4,
                            facecolor=facecolor, alpha=0.85,
                            edgecolor="black", linewidth=1.5)
        ax.add_patch(circle)
        label = f"{region_info['name']}\n{change:.1f}×" if pd.notna(change) else region_info["name"]
        ax.annotate(label, (region_info["lon"], region_info["lat"]),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlim(-15, 45)
    ax.set_ylim(33, 72)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Compound Heat-Drought: Regional Change Factors (SSP5-8.5)",
                 fontweight="bold", fontsize=12)
    ax.set_aspect("equal")

    legend_elements = [
        mpatches.Patch(facecolor="#B2182B", edgecolor="black", label="≥2× increase"),
        mpatches.Patch(facecolor="#F4A582", edgecolor="black", label="1.5–2× increase"),
        mpatches.Patch(facecolor="#FDDBC7", edgecolor="black", label="1–1.5× increase"),
        mpatches.Patch(facecolor="#92C5DE", edgecolor="black", label="Decrease"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True, fontsize=9, fancybox=True)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 -- Return periods (was fig4_return_periods)
# ---------------------------------------------------------------------------

def _rename_event_label(name):
    """Use 'Greece Heat-Drought' in figure labels (file column stays as-is)."""
    return name.replace("Greece Fire Weather", "Greece Heat-Drought")


def plot_return_periods(table4b):
    if len(table4b) == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    events = table4b["Event"].tolist()
    display_events = [_rename_event_label(e) for e in events]
    x = np.arange(len(events))
    width = 0.2
    labels = ["Historical", "SSP1-2.6", "SSP3-7.0", "SSP5-8.5"]
    colors = [COLORS["historical"], COLORS["ssp126"], COLORS["ssp370"], COLORS["ssp585"]]

    ax = axes[0]
    rate_cols = ["Hist_Rate_Per_Decade", "SSP126_Rate", "SSP370_Rate", "SSP585_Rate"]
    for i, (col, label, color) in enumerate(zip(rate_cols, labels, colors)):
        ax.bar(x + (i - 1.5) * width, table4b[col].values, width,
               label=label, color=color, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Events per decade")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace(" ", "\n") for e in display_events], fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("A. Event Frequency", fontweight="bold", loc="left")
    for i, event in enumerate(events):
        row = table4b[table4b["Event"] == event].iloc[0]
        change = row["SSP585_Change_Factor"]
        if pd.notna(change) and change != np.inf:
            color = COLORS["ssp585"] if change > 1.2 else ("#404040" if change < 0.8 else "#666666")
            y_pos = row["SSP585_Rate"] + 2
            direction = "↑" if change > 1 else "↓"
            ax.annotate(f"{change:.1f}× {direction}", xy=(i + 1.5 * width, y_pos),
                        ha="center", fontsize=9, fontweight="bold", color=color)

    ax = axes[1]
    return_cols = ["Hist_Return_Years", "SSP126_Return_Years", "SSP370_Return_Years", "SSP585_Return_Years"]
    for i, (col, label, color) in enumerate(zip(return_cols, labels, colors)):
        values = np.clip(table4b[col].values.copy(), 0, 15)
        ax.bar(x + (i - 1.5) * width, values, width,
               label=label, color=color, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Return period (years)")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace(" ", "\n") for e in display_events], fontsize=9)
    ax.set_title("B. Return Periods", fontweight="bold", loc="left")
    ax.set_ylim(0, 10)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(events) - 0.3, 1.2, "Annual", fontsize=8, color="gray", style="italic")

    if "Greece Fire Weather" in events:
        gi = events.index("Greece Fire Weather")
        row = table4b[table4b["Event"] == "Greece Fire Weather"].iloc[0]
        ax.annotate("",
                    xy=(gi + 1.5 * width, row["SSP585_Return_Years"]),
                    xytext=(gi - 1.5 * width, row["Hist_Return_Years"]),
                    arrowprops=dict(arrowstyle="->", color="#B2182B", lw=2))
        ax.annotate(f"{row['SSP585_Change_Factor']:.0f}× more\nfrequent",
                    xy=(gi + 0.5, 3.5), ha="center", fontsize=9, fontweight="bold", color="#B2182B")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6 -- Validation map (was fig5_validation_map)
# ---------------------------------------------------------------------------

def plot_validation_map(table4b):
    fig, ax = plt.subplots(figsize=(10, 8))
    europe = _load_europe_boundaries()
    if europe is not None:
        europe.plot(ax=ax, color="#f5f5f5", edgecolor="gray", linewidth=0.5)
    else:
        ax.set_facecolor("#e6f2ff")

    events_info = {
        "Vaia Windstorm": {
            "lat": 42.9, "lon": 2.4, "color": COLORS["windstorm"],
            "primary": "Surface Pressure\n(Cyclone)", "conditioning": "—",
            "date": "Oct 2018", "lookup_name": "Vaia Windstorm",
        },
        "Western Europe Floods": {
            "lat": 48.9, "lon": 5.8, "color": COLORS["flood"],
            "primary": "Precipitation", "conditioning": "Runoff +\nSoil Moisture",
            "date": "Jul 2021", "lookup_name": "Western Europe Floods",
        },
        "Greece Heat-Drought": {
            "lat": 41.5, "lon": 24.6, "color": COLORS["heat_drought_fire"],
            "primary": "Max Temperature", "conditioning": "Soil Moisture\n(Dry)",
            "date": "Aug 2021", "lookup_name": "Greece Fire Weather",
        },
    }

    for label, info in events_info.items():
        row = table4b[table4b["Event"] == info["lookup_name"]] if len(table4b) > 0 else pd.DataFrame()
        if len(row) > 0 and pd.notna(row.iloc[0]["SSP585_Change_Factor"]):
            info["signal"] = f"{row.iloc[0]['SSP585_Change_Factor']:.1f}×"
        else:
            info["signal"] = "—"

    for event_name, info in events_info.items():
        ax.scatter(info["lon"], info["lat"], s=400, c=info["color"],
                   edgecolor="black", linewidth=2, zorder=5, marker="o")
        if "Vaia" in event_name:
            offset = (-8, 3); ha = "center"
        elif "Western" in event_name:
            offset = (-5, 5); ha = "center"
        else:
            offset = (5, 3); ha = "center"
        ax.annotate(f"{event_name}\n{info['date']}",
                    xy=(info["lon"], info["lat"]),
                    xytext=(info["lon"] + offset[0], info["lat"] + offset[1]),
                    fontsize=10, fontweight="bold", ha=ha,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=info["color"], linewidth=2, alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color=info["color"], lw=1.5))
        detail = f"Primary: {info['primary']}\nConditioning: {info['conditioning']}\nFuture: {info['signal']}"
        ax.annotate(detail,
                    xy=(info["lon"] + offset[0], info["lat"] + offset[1] - 2.5),
                    fontsize=8, ha=ha, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9",
                              edgecolor="gray", linewidth=0.5, alpha=0.9))

    ax.set_xlim(-12, 40)
    ax.set_ylim(35, 62)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Validated Compound Events: Detection & Future Projections",
                 fontweight="bold", fontsize=13)
    ax.set_aspect("equal")

    legend_elements = [
        mpatches.Patch(facecolor=COLORS["windstorm"], edgecolor="black", label="Compound Windstorm"),
        mpatches.Patch(facecolor=COLORS["flood"], edgecolor="black", label="Compound Flood"),
        mpatches.Patch(facecolor=COLORS["heat_drought_fire"], edgecolor="black", label="Compound Heat-Drought"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True, fontsize=9,
              title="Event Type", title_fontsize=10)

    if len(table4b) > 0 and "Greece Fire Weather" in table4b["Event"].values:
        gr = table4b[table4b["Event"] == "Greece Fire Weather"].iloc[0]
        textstr = (
            f"Key Finding:\nGreece-2021-like events\n"
            f"1 in {gr['Hist_Return_Years']:.0f}yr → 1 in {gr['SSP585_Return_Years']:.1f}yr\n"
            f"({gr['SSP585_Change_Factor']:.0f}× more frequent)"
        )
    else:
        textstr = "Key Finding:\nValidation data unavailable"
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="#fff3e6",
                      edgecolor=COLORS["heat_drought_fire"], linewidth=2, alpha=0.95),
            fontweight="bold")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    setup_style()
    data = load_data()

    print("\nFigure 1: pipeline schematic")
    fig1 = plot_pipeline_schematic()
    save_figure(fig1, "fig1_pipeline")

    print("\nFigure 2: frequency")
    table1 = analyze_frequency(data)
    save_figure(plot_frequency(table1), "fig2_frequency")

    print("\nFigure 3: extremes")
    table2 = analyze_characteristics(data)
    save_figure(plot_extremes(table2), "fig3_extremes")

    print("\nFigure 4: regional map")
    table3 = analyze_regional(data)
    save_figure(plot_regional(table3), "fig4_regional_map")

    print("\nFigure 5: return periods")
    table4a, table4b = analyze_validation(data)
    fig5 = plot_return_periods(table4b)
    if fig5 is not None:
        save_figure(fig5, "fig5_return_periods")

    print("\nFigure 6: validation map")
    save_figure(plot_validation_map(table4b), "fig6_validation_map")

    print("\nDone.")


if __name__ == "__main__":
    main()
