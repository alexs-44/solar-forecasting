"""
data_loader.py
=============
Code writing assisted by Claude AI
=============
Loads real NREL NSRDB solar irradiance and meteorological data for 5 North
American sites across 3 years (2017, 2018, 2019), computes AC power output
using a physics-based PV model, and saves a combined dataset to
data/pvdaq_combined.csv.

Data source:
    NREL National Solar Radiation Database (NSRDB)
    Downloaded from: nsrdb.nlr.gov/data-viewer
    Resolution: hourly, years 2017-2019

Sites:
    Phoenix, AZ   — Hot Desert
    Denver, CO    — Cold Continental
    Seattle, WA   — Marine/Cloudy
    Miami, FL     — Humid Subtropical
    Boston, MA    — Temperate/Northern

PV system modelled:
    Capacity : 5 kW, Module eff: 18%, Temp coeff: -0.004/°C
    NOCT: 44°C, Inverter eff: 96%

Usage:
    python src/data_loader.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ── Site definitions — 3 years per site ──────────────────────────────────────
SITES = [
    {"files": ["Phoenix_2017.csv", "Phoenix_2018.csv", "Phoenix_2019.csv"],
     "site_id": 1, "name": "Phoenix_AZ",
     "climate": "Hot Desert",        "lat": 33.45, "lon": -112.07},
    {"files": ["Denver_2017.csv",  "Denver_2018.csv",  "Denver_2019.csv"],
     "site_id": 2, "name": "Denver_CO",
     "climate": "Cold Continental",  "lat": 39.74, "lon": -104.98},
    {"files": ["Seattle_2017.csv", "Seattle_2018.csv", "Seattle_2019.csv"],
     "site_id": 3, "name": "Seattle_WA",
     "climate": "Marine",            "lat": 47.61, "lon": -122.33},
    {"files": ["Miami_2017.csv",   "Miami_2018.csv",   "Miami_2019.csv"],
     "site_id": 4, "name": "Miami_FL",
     "climate": "Humid Subtropical", "lat": 25.77, "lon": -80.19},
    {"files": ["Boston_2017.csv",  "Boston_2018.csv",  "Boston_2019.csv"],
     "site_id": 5, "name": "Boston_MA",
     "climate": "Temperate",         "lat": 42.36, "lon": -71.06},
]

# ── PV system parameters ──────────────────────────────────────────────────────
SYSTEM_CAPACITY_W = 5000
EFF_REF           = 0.18
TEMP_COEFF        = -0.004
T_REF             = 25.0
NOCT              = 44.0
INV_EFF           = 0.96
GHI_NIGHT_THRESH  = 50
TARGET_PER_SITE   = 6000   # ~2000 per year × 3 years
DATA_DIR          = Path("data")
OUTPUT_PATH       = DATA_DIR / "pvdaq_combined.csv"


def compute_poa(ghi, zenith, tilt=20.0):
    cos_zenith = np.cos(np.radians(zenith.clip(0, 89)))
    cos_poa    = np.cos(np.radians(zenith.clip(0, 89) - tilt))
    ratio      = np.where(cos_zenith > 0.01, cos_poa / cos_zenith, 1.0)
    return pd.Series((ghi * ratio).clip(lower=0), index=ghi.index)


def compute_module_temp(t_ambient, ghi):
    return t_ambient + (NOCT - 20) * ghi / 800.0


def compute_ac_power(poa, t_module, rng):
    eta   = (EFF_REF * (1 + TEMP_COEFF * (t_module - T_REF))).clip(lower=0)
    p_dc  = poa * eta * (SYSTEM_CAPACITY_W / (EFF_REF * 1000))
    noise = 1 + rng.normal(0, 0.02, size=len(p_dc))
    return (p_dc * INV_EFF * noise).clip(lower=0, upper=SYSTEM_CAPACITY_W)


def load_one_file(filepath, site, rng):
    """Load and process a single NSRDB CSV file."""
    if not filepath.exists():
        print(f"    [SKIP] File not found: {filepath.name}")
        return None

    df = pd.read_csv(filepath, skiprows=2)
    df = df.rename(columns={
        "Temperature":        "t_ambient",
        "GHI":                "ghi",
        "DHI":                "dhi",
        "DNI":                "dni",
        "Wind Speed":         "wind_speed",
        "Solar Zenith Angle": "zenith",
        "Relative Humidity":  "humidity",
    })

    df["timestamp"] = pd.to_datetime(dict(
        year=df["Year"], month=df["Month"],
        day=df["Day"],   hour=df["Hour"]
    ))
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["poa_irradiance"] = compute_poa(df["ghi"], df["zenith"])
    df["t_module"]       = compute_module_temp(df["t_ambient"], df["ghi"])
    df["ac_power_w"]     = compute_ac_power(df["poa_irradiance"],
                                            df["t_module"], rng)

    # Filter nighttime
    df = df[df["ghi"] >= GHI_NIGHT_THRESH].copy()

    if len(df) < 50:
        print(f"    [SKIP] Too few daytime rows in {filepath.name}: {len(df)}")
        return None

    df["site_id"]   = site["site_id"]
    df["site_name"] = site["name"]
    df["climate"]   = site["climate"]
    df["lat"]       = site["lat"]
    df["lon"]       = site["lon"]
    df["hour"]      = df["timestamp"].dt.hour
    df["doy"]       = df["timestamp"].dt.dayofyear
    df["month"]     = df["timestamp"].dt.month

    keep = [
        "timestamp", "site_id", "site_name", "climate",
        "lat", "lon", "hour", "doy", "month",
        "ghi", "poa_irradiance", "t_ambient", "t_module",
        "wind_speed", "ac_power_w",
    ]
    return df[keep]


def load_site(site, rng):
    """Load and concatenate all years for one site."""
    frames = []
    for filename in site["files"]:
        path = DATA_DIR / filename
        df   = load_one_file(path, site, rng)
        if df is not None:
            frames.append(df)
            print(f"    Loaded {filename}: {len(df):,} daytime rows")

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def main():
    print("=" * 60)
    print("  NSRDB Solar Data Loader  (3 years × 5 sites)")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    all_frames = []
    for site in SITES:
        print(f"\n  Site {site['site_id']} — {site['name']} ({site['climate']})")
        df = load_site(site, rng)
        if df is None:
            continue

        print(f"    Total daytime rows : {len(df):,}")
        print(f"    Power range        : {df['ac_power_w'].min():.0f}"
              f" – {df['ac_power_w'].max():.0f} W")
        print(f"    Mean AC power      : {df['ac_power_w'].mean():.0f} W")

        # Subsample evenly preserving temporal order
        if len(df) > TARGET_PER_SITE:
            step = len(df) // TARGET_PER_SITE
            df   = df.iloc[::step].head(TARGET_PER_SITE).copy()
        print(f"    After subsample    : {len(df):,}")
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No data loaded. Check all CSV files are in data/")

    combined = (pd.concat(all_frames, ignore_index=True)
                  .sort_values(["site_id", "timestamp"])
                  .reset_index(drop=True))

    # Inject 2% missing values to simulate sensor dropout
    for col in ["t_ambient", "wind_speed"]:
        mask = rng.random(len(combined)) < 0.02
        combined.loc[mask, col] = np.nan

    combined.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print(f"  Saved → {OUTPUT_PATH}")
    print(f"  Total samples : {len(combined):,}")
    print(f"\n  Site breakdown:")
    for sid, grp in combined.groupby("site_id"):
        print(f"    {grp['site_name'].iloc[0]:<20} {len(grp):>5,} rows"
              f"  |  mean = {grp['ac_power_w'].mean():.0f} W")
    print("=" * 60)
    print("\n  Next: python src/train.py")


if __name__ == "__main__":
    main()
