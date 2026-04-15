"""
preprocessing.py
===============
Code writing assisted by Claude AI
================

Feature engineering and preprocessing pipeline.

TRUE FORECASTING MODEL:
  - Target is now y(t + HORIZON) instead of y(t)
  - make_sequences() accepts a `horizon` parameter (default 1 = 1 hour ahead)
  - Sequences never cross site boundaries
  - Linear Regression uses a flattened 24-hour window as its feature vector
    so it is trained on the same information as the deep models

Why this matters:
  Predicting current power from current weather is trivially easy (linear).
  Predicting future power from past weather patterns is the real operational
  problem as grid operators need advance notice of generation, not instant
  readings. With a forecast horizon, LSTM and CNN-LSTM are now the appropriate
  tools because they can learn temporal patterns (e.g. morning cloud clearing,
  afternoon temperature peaks) that persist into future time steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

DATA_PATH    = Path("data/nsrdb_combined.csv")
RESULTS_DIR  = Path("results")
RANDOM_STATE = 42

# Default forecast horizon: predict power this many hours ahead
DEFAULT_HORIZON = 1

METEO_COLS = [
    "ghi",
    "poa_irradiance",
    "t_ambient",
    "t_module",
    "wind_speed",
]
TARGET_COL = "ac_power_w"


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    return df


def impute_missing(df, cols):
    df = df.copy()
    for sid in df["site_id"].unique():
        mask = df["site_id"] == sid
        for col in cols:
            if col in df.columns and df.loc[mask, col].isna().any():
                median_val = df.loc[mask, col].median()
                df.loc[mask & df[col].isna(), col] = median_val
    return df


def add_engineered_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df["doy"]  / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * df["doy"]  / 365)
    if "lat" not in df.columns:
        df["lat"] = 0.0
    if "lon" not in df.columns:
        df["lon"] = 0.0
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    Add lagged power output and GHI rate-of-change as features.

    Why this helps:
      Recent power output is the single best predictor of near-future power.
      If the system produced 3,000W an hour ago and 3,200W just now, that
      upward trend carries real information about what will happen next.
      Without lag features, the model has no direct knowledge of recent
      system state, and will know only weather inputs.

    Features added:
      power_lag_1, _2, _3 : actual power output 1, 2, 3 hours ago
      ghi_lag_1            : GHI 1 hour ago (cloud persistence signal)
      ghi_delta            : GHI change from last hour (trending up or down?)

    Lags are computed per-site to avoid bleeding across site boundaries.
    Rows where any lag is undefined (start of each site's sequence) are dropped.
    """
    df = df.copy()

    # Per-site lag computation to never bleed values across site boundaries
    for lag in lags:
        df[f"power_lag_{lag}"] = (
            df.groupby("site_id")["ac_power_w"].shift(lag)
        )

    # GHI lag and rate of change
    df["ghi_lag_1"] = df.groupby("site_id")["ghi"].shift(1)
    df["ghi_delta"] = df.groupby("site_id")["ghi"].diff()   # change from last hour

    # Drop rows where any lag feature is NaN (start of each site's series)
    lag_cols = [f"power_lag_{l}" for l in lags] + ["ghi_lag_1", "ghi_delta"]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    return df


def build_feature_matrix(df):
    """
    Returns both the feature matrix AND the cleaned full DataFrame.
    The cleaned df has lag-NaN rows dropped and is needed to rebuild
    y and site_id arrays that align with X_df after the drop.
    """
    df = add_engineered_features(df)
    df = add_lag_features(df, lags=[1, 2, 3])   # drops ~3 rows per site
    df = impute_missing(df, METEO_COLS)
    df = df.reset_index(drop=True)               # clean 0-based index

    feature_cols = METEO_COLS + [
        "hour_sin", "hour_cos",
        "doy_sin",  "doy_cos",
        "lat", "lon",
        "power_lag_1", "power_lag_2", "power_lag_3",
        "ghi_lag_1", "ghi_delta",
    ]
    return df[feature_cols], feature_cols, df     # also return cleaned df


def load_and_preprocess(
    data_path=DATA_PATH,
    split="time",
    test_site_id=None,
    test_fraction=0.20,
    save_scaler=True,
):
    df_raw = load_raw(data_path)

    if test_site_id is None:
        test_site_id = int(df_raw["site_id"].max())

    # build_feature_matrix drops lag-NaN rows and returns the cleaned df
    # so ALL subsequent operations use df (not df_raw) for alignment
    X_df, feature_cols, df = build_feature_matrix(df_raw)
    y           = df[TARGET_COL].values.astype(float)
    site_id_arr = df["site_id"].values

    if split == "time":
        masks = []
        for sid in np.unique(site_id_arr):
            idx = np.where(site_id_arr == sid)[0]
            cut = int(len(idx) * (1 - test_fraction))
            t   = np.zeros(len(df), bool)
            t[idx[cut:]] = True
            masks.append(t)
        test_mask  = np.any(masks, axis=0)
        train_mask = ~test_mask
    elif split == "site":
        test_mask  = site_id_arr == test_site_id
        train_mask = ~test_mask
    else:
        raise ValueError(f"split must be 'time' or 'site', got '{split}'")

    X_train_raw = X_df.values[train_mask]
    X_test_raw  = X_df.values[test_mask]
    y_train     = y[train_mask]
    y_test      = y[test_mask]

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    if save_scaler:
        RESULTS_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler,       RESULTS_DIR / "scaler.pkl")
        joblib.dump(feature_cols, RESULTS_DIR / "feature_cols.pkl")

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    print(f"  Preprocessing done  |  split='{split}'")
    print(f"  Train : {X_train.shape[0]:,} rows  |  features: {X_train.shape[1]}")
    print(f"  Test  : {X_test.shape[0]:,} rows")

    return (X_train, y_train, X_test, y_test,
            scaler, feature_cols, df_train, df_test)


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window: int = 30,
    horizon: int = DEFAULT_HORIZON,
    site_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for horizon-ahead forecasting.

    INPUT  : X[i : i+window]          — past `window` hours of weather
    TARGET : y[i + window + horizon-1] — power `horizon` hours after window ends

    With horizon > 0, models must learn from temporal patterns to anticipate
    future conditions — making LSTM and CNN architectures meaningful.

    Parameters
    ----------
    window   : look-back window in hours (default 24)
    horizon  : hours ahead to predict    (default 1)
    site_ids : prevents sequences crossing site boundaries
    """
    if site_ids is None:
        site_ids = np.zeros(len(X), int)

    X_list, y_list = [], []
    for sid in np.unique(site_ids):
        idx    = np.where(site_ids == sid)[0]
        Xs, ys = X[idx], y[idx]
        for i in range(len(Xs) - window - horizon + 1):
            X_list.append(Xs[i : i + window])           # shape: (window, features)
            y_list.append(ys[i + window + horizon - 1]) # scalar: future power

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.float32))


def make_flat_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window: int = 30,
    horizon: int = DEFAULT_HORIZON,
    site_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as make_sequences but flattens window → 1D vector for Linear Regression.
    Ensures LR is trained on the same past-24h information as the deep models,
    making the comparison fair.

    Returns shape: (n_sequences, window * n_features)
    """
    X_seq, y_seq = make_sequences(X, y, window, horizon, site_ids)
    n, w, f = X_seq.shape
    return X_seq.reshape(n, w * f), y_seq
