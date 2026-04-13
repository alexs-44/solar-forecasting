"""
app.py — SolCast Forecasting Interface 
=============================================
User enters latitude, longitude, and selects a forecast date.
The app fetches a real hourly weather forecast from the Open-Meteo API
and predicts solar PV power output HORIZON hours into the future.


Run:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"

st.set_page_config(
    page_title="SolCast",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Anton:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1f2d 50%, #0a1628 100%);
    border: 1px solid #1a3a5c; border-radius: 16px;
    padding: 32px 40px; margin-bottom: 24px;
}
.hero h1 {
    font-family: 'Anton', sans-serif; font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #f5a623, #f7c948, #fff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}
.hero p { color: #7a9ab5; font-size: 0.95rem; margin: 0; }
.forecast-badge {
    background: #0d2a1a; border: 1px solid #3fb950;
    border-radius: 8px; padding: 10px 16px;
    color: #3fb950; font-size: 0.85rem; margin: 8px 0;
    display: inline-block;
}
.metric-card {
    background: #0d1f2d; border: 1px solid #1a3a5c;
    border-radius: 12px; padding: 18px 20px;
    text-align: center; margin: 6px 0;
}
.metric-card .model-label {
    color: #7a9ab5; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;
}
.metric-card .power-val { color: #f5f5f5; font-size: 2rem; font-weight: 500; }
.metric-card .power-sub { color: #f5a623; font-size: 0.8rem; margin-top: 4px; }
.info-box {
    background: #0d1f2d; border-left: 3px solid #f5a623;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    margin: 12px 0; color: #c9d9e8; font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    models = {}
    try:
        models["Linear Regression"] = joblib.load(RESULTS_DIR / "linear_regression.pkl")
    except FileNotFoundError:
        pass
    try:
        from tensorflow import keras
        for name, key in [("LSTM","lstm"), ("CNN1D","cnn1d"), ("CNN-LSTM","cnn_lstm")]:
            ckpt = RESULTS_DIR / f"{key}_best.keras"
            if ckpt.exists():
                models[name] = keras.models.load_model(str(ckpt))
    except Exception:
        pass
    return models


@st.cache_resource(show_spinner="Loading scaler…")
def load_scaler():
    try:
        scaler    = joblib.load(RESULTS_DIR / "scaler.pkl")
        feat_cols = joblib.load(RESULTS_DIR / "feature_cols.pkl")
        return scaler, feat_cols
    except FileNotFoundError:
        return None, None


def load_config():
    cfg_path = RESULTS_DIR / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {"horizon": 1, "window": 24}


@st.cache_data(ttl=3600, show_spinner="Fetching weather forecast…")
def fetch_weather(lat, lon, forecast_date):
    """Fetch hourly weather from Open-Meteo (free, no API key)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":       lat,
        "longitude":      lon,
        "hourly":         "shortwave_radiation,temperature_2m,windspeed_10m",
        "start_date":     forecast_date,
        "end_date":       forecast_date,
        "timezone":       "auto",
        "windspeed_unit": "ms",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()["hourly"]
        return pd.DataFrame({
            "hour":       range(24),
            "ghi":        data["shortwave_radiation"],
            "t_ambient":  data["temperature_2m"],
            "wind_speed": data["windspeed_10m"],
        })
    except Exception as e:
        st.error(f"Weather API error: {e}")
        return None


def build_features(weather_df, lat, lon, forecast_date, feat_cols):
    """Build feature matrix matching training schema."""
    doy = pd.Timestamp(forecast_date).dayofyear
    df  = weather_df.copy()
    df["poa_irradiance"] = df["ghi"] * 1.05
    df["t_module"]       = df["t_ambient"] + (44 - 20) * df["ghi"] / 800
    df["hour_sin"]       = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]       = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]        = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]        = np.cos(2 * np.pi * doy / 365)
    df["lat"]            = lat
    df["lon"]            = lon
    X = np.array([[row.get(c, 0.0) for c in feat_cols]
                  for _, row in df.iterrows()])
    return X


MODEL_COLORS = {
    "Linear Regression": "#4C72B0",
    "LSTM":              "#55A868",
    "CNN1D":             "#DD8452",
    "CNN-LSTM":          "#C44E52",
}


def predict_day(models, scaler, feat_cols, X_raw, window, horizon, weather_df):
    """
    Run all models for a full 24-hour day.

    For each target hour h, the input is the window of hours ending at h-horizon.
    This correctly simulates the forecasting scenario:
        "What will power be at hour h, given weather up to hour h-horizon?"
    """
    X_scaled = scaler.transform(X_raw)
    results  = {}

    for name, model in models.items():
        preds = np.zeros(24)
        for h in range(24):
            # Build the input window ending `horizon` hours before target
            end   = max(0, h - horizon + 1)
            start = max(0, end - window)
            window_data = X_scaled[start:end] if end > start else X_scaled[:1]

            pad = window - len(window_data)
            if pad > 0:
                window_data = np.vstack([
                    np.zeros((pad, X_scaled.shape[1])), window_data
                ])

            if name == "Linear Regression":
                flat = window_data.reshape(1, -1)
                val  = float(model.predict(flat)[0])
            else:
                seq = window_data.reshape(1, window, -1).astype(np.float32)
                val = float(model.predict(seq, verbose=0).flatten()[0])

            preds[h] = max(0, val)
        # Zero out nighttime hours where GHI is below generation threshold
        night_mask = weather_df["ghi"].values < 50
        preds[night_mask] = 0.0
        results[name] = preds
    return results


def make_forecast_plot(predictions, weather_df, horizon):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1f2d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0a1628")
        ax.tick_params(colors="#7a9ab5")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a3a5c")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    hours  = weather_df["hour"].values
    styles = ["-", "--", "-.", ":"]
    for (name, preds), ls in zip(predictions.items(), styles):
        ax1.plot(hours, preds / 1000, label=name,
                 color=MODEL_COLORS.get(name, "#aaa"),
                 lw=2, ls=ls, marker="o", ms=3, alpha=0.9)

    ax1.fill_between(
        hours,
        np.min([p for p in predictions.values()], axis=0) / 1000,
        np.max([p for p in predictions.values()], axis=0) / 1000,
        alpha=0.08, color="#f5a623",
    )
    ax1.set_ylabel("Predicted AC Power (kW)", color="#c9d9e8", fontsize=11)
    ax1.set_title(
        f"Solar Power Forecast — Hourly 1h-Ahead Predictions for {date_str}",
        color="#f5f5f5", fontsize=13, fontweight="bold", pad=10,
    )
    ax1.legend(fontsize=9, facecolor="#0d1f2d",
               labelcolor="#c9d9e8", edgecolor="#1a3a5c")
    ax1.set_xlim(0, 23)

    ax2.bar(hours, weather_df["ghi"], color="#f5a623", alpha=0.5, width=0.8)
    ax2.set_xlabel("Hour of Day", color="#c9d9e8", fontsize=10)
    ax2.set_ylabel("GHI (W/m²)", color="#7a9ab5", fontsize=9)
    ax2.set_xlim(0, 23)

    plt.tight_layout(h_pad=2)
    return fig


def main():
    cfg     = load_config()
    horizon = cfg["horizon"]
    window  = cfg["window"]

    st.markdown(f"""
    <div class="hero">
        <h1> SolCast <span style="font-weight:400; font-style:italic; font-size:1.8rem;">— Solar Output Forecasting</span></h1>
        <p>Enter a location and date to get a <strong>{horizon}-hour ahead</strong>
        solar power forecast driven by real weather data.</p>
    </div>
    <div class="forecast-badge">
        Forecasting {horizon} hour(s) ahead using past {window}-hour weather window
    </div>
    """, unsafe_allow_html=True)

    models        = load_models()
    scaler, feat_cols = load_scaler()

    if not models or scaler is None:
        st.error("ERROR: No trained models found. Run `python src/train.py` first.")
        st.stop()

    st.markdown(
        f'<div class="info-box"> {len(models)} model(s) loaded: '
        f'{", ".join(models.keys())}</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header(" Location & Date")
        preset = st.selectbox("Quick-pick a site", [
            "Custom",
            "Phoenix, AZ (Hot Desert)",
            "Denver, CO (Cold Continental)",
            "Seattle, WA (Marine)",
            "Miami, FL (Humid Subtropical)",
            "Boston, MA (Temperate)",
        ])
        presets = {
            "Phoenix, AZ (Hot Desert)":      (33.45, -112.07),
            "Denver, CO (Cold Continental)": (39.74, -104.98),
            "Seattle, WA (Marine)":          (47.61, -122.33),
            "Miami, FL (Humid Subtropical)": (25.77,  -80.19),
            "Boston, MA (Temperate)":        (42.36,  -71.06),
        }
        default_lat, default_lon = presets.get(preset, (37.77, -122.42))
        lat = st.number_input("Latitude",  value=default_lat,
                              min_value=-90.0,  max_value=90.0,  step=0.01)
        lon = st.number_input("Longitude", value=default_lon,
                              min_value=-180.0, max_value=180.0, step=0.01)
        st.markdown("---")
        forecast_date = st.date_input(
            "Forecast Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=6),
        )
        st.markdown("---")
        run_btn = st.button("Run Forecast", use_container_width=True,
                            type="primary")

    # ── Main panel ────────────────────────────────────────────────────────────
    if run_btn:
        date_str = forecast_date.strftime("%Y-%m-%d")
        with st.spinner("Fetching weather from Open-Meteo…"):
            weather = fetch_weather(lat, lon, date_str)
        if weather is None:
            st.stop()

        with st.spinner("Running forecast models…"):
            X_raw       = build_features(weather, lat, lon, date_str, feat_cols)
            predictions = predict_day(models, scaler, feat_cols,
                          X_raw, window, horizon, weather)

        st.subheader(
            f"{horizon}-24-Hour Forecast for — {date_str}  |  "
            f"{lat:.3f}°N, {lon:.3f}°E"
        )

        st.markdown(f"""
        <div class="info-box">
        NOTE: These predictions show estimated power output <strong>{horizon} hour(s)
        from each time step</strong>, based on weather patterns in the preceding
        {window} hours. This is a forecast rather than an instantaneous reading.
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(len(predictions))
        for col, (name, preds) in zip(cols, predictions.items()):
            peak  = preds.max()
            total = preds.sum() / 1000
            col.markdown(f"""
            <div class="metric-card">
                <div class="model-label">{name}</div>
                <div class="power-val">{peak/1000:.2f} kW</div>
                <div class="power-sub">Daily yield: {total:.1f} kWh</div>
            </div>
            """, unsafe_allow_html=True)

        fig = make_forecast_plot(predictions, weather, horizon)
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Hourly Predictions (W)"):
            table = pd.DataFrame({"Hour": range(24)})
            for name, preds in predictions.items():
                table[name] = preds.round(1)
            table["GHI (W/m²)"] = weather["ghi"].round(1)
            st.dataframe(table, hide_index=True, use_container_width=True)

    else:
        st.markdown("""
        <div class="info-box">
        Select a location and date in the sidebar, then click
        <strong>Run Forecast</strong>.
        </div>
        """, unsafe_allow_html=True)
        comp = RESULTS_DIR / "plots" / "model_comparison.png"
        if comp.exists():
            st.subheader("Training Results")
            st.image(str(comp), caption="Model Comparison on Forecasting Test Set")

    st.markdown("---")
    with st.expander("Why forecasting instead of instantaneous prediction?"):
        st.markdown(f"""
**The problem with predicting current power from current weather:**
A simple linear equation (Power ≈ 0.18 × GHI × area) explains >99% of variance.
There's nothing for a neural network to learn and Linear Regression wins trivially.

**The real operational need:**
Grid operators and building managers need to know generation *in advance* — not
right now. "Will my solar system produce enough power at 3pm to avoid buying
from the grid?" requires a forecast, not a reading.

**Why LSTM and CNN-LSTM are appropriate for forecasting:**
- LSTMs learn temporal dependencies: if the past few hours show increasing
  irradiance, that pattern often continues.
- CNNs detect recurring local patterns: morning cloud clearing, afternoon peaks.
- Linear Regression cannot model these sequential relationships.

**This model:** Uses the past **{window} hours** of weather data to predict power
**{horizon} hour(s) ahead** — a genuine forecasting task where sequence models
have a real advantage over linear baselines.
        """)


if __name__ == "__main__":
    main()
