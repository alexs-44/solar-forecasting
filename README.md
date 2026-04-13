# SolCast — Solar PV Power Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Data](https://img.shields.io/badge/Data-NREL%20NSRDB-yellow)


> **Predict the AC power output of a solar PV system for any location on Earth.
> Simply enter a latitude, longitude, and date, and SolCast fetches a real weather
> forecast and runs four ML models to give you an hourly power prediction.**

Trained on real measured data from the [NSRDB](https://nsrdb.nlr.gov/data-viewer) database
across five North American climate zones. Compares Linear Regression, LSTM,
1D-CNN, and a CNN-LSTM hybrid.

---

## Table of Contents

1. [Project Objective](#-project-objective)
2. [Key Results](#-key-results)
3. [How It Works](#-how-it-works)
4. [Repository Structure](#-repository-structure)
5. [Dataset](#-dataset)
6. [Model Architectures](#-model-architectures)
7. [Quick Start — Local](#-quick-start--local)
8. [Running the App](#-running-the-app)
9. [Reproducing Results](#-reproducing-results)
10. [Limitations](#-limitations)
11. [Weather API](#-weather-api)
12. [Configuration](#-configuration)
13. [Troubleshooting](#-troubleshooting)
14. [Author](#-author)

---

## Project Objective

Solar photovoltaic systems are the fastest-growing source of electricity worldwide,
but their output varies with weather conditions in ways that are hard to predict
without the right tools. This project builds an end-to-end machine learning pipeline
that answers a simple question:

> **Given a location and a weather forecast, how much power will a 5 kW solar
> system produce tomorrow?**

The pipeline covers everything from raw data ingestion through model training,
evaluation, and a live interactive web application.

### What this project demonstrates

- Downloading and preprocessing real measured irradiance data from the NREL NSRDB database
- Engineering physically meaningful features (cyclical time encodings,
  geographic coordinates, NOCT-based module temperature)
- Training and comparing four model architectures on the same dataset
- Building a production-style inference app backed by a live weather API

---

## Key Results

Results on the held-out 20% test set (1-hour ahead forecasting, 3 years × 5 sites):

| Model | RMSE (W) ↓ | MAE (W) ↓ | R² ↑ |
|---|---|---|---|
| Linear Regression | 574 | 409 | 0.742 |
| **LSTM** | **526** | **359** | **0.784** |
| 1D-CNN | 538 | 375 | 0.774 |
| **CNN-LSTM** | **527** | 374 | **0.783** |

LSTM and CNN-LSTM both outperform the linear baseline, confirming that 
sequential models provide meaningful gains for the 1-hour ahead forecasting task.

**Instantaneous prediction baseline** (for reference):

| Model | R² |
|---|---|
| Linear Regression | 0.995 |
| LSTM | 0.714 |

Linear Regression dominated instantaneous prediction due to the near-linear 
irradiance-power relationship, motivating the shift to a true forecasting formulation.

> The CNN-LSTM hybrid achieves the best overall accuracy across all metrics and
> generalises well to an unseen geographic location, confirming the value of using
> continuous lat/lon features instead of one-hot site encodings.

**Key Findings:**
- LSTM and CNN-LSTM outperform Linear Regression under the forecasting 
  formulation (R²=0.784 vs 0.742)
- Reducing dropout from 0.2 to 0.1 improved CNN1D from R²=0.724 to R²=0.774
- Expanding training data from 1 to 3 years improved LSTM R² by 0.068 points
- Instantaneous prediction is dominated by Linear Regression (R²=0.995) — 
  reformulating as forecasting is necessary to show deep learning value
---

## How It Works

### End-to-end pipeline

```
NREL NSRDB (manual CSV download)
│
▼
src/data_loader.py          ← loads and processes real NSRDB data for 5 sites
│
▼
src/preprocessing.py        ← feature engineering, lag features, train/test splits
│
▼
src/train.py                ← trains all 4 models, saves artefacts
│
▼
src/evaluate.py             ← computes RMSE/MAE/R², generates all plots
│
▼
app/app.py                  ← Streamlit app: user enters lat/lon + date
→ Open-Meteo API fetches real forecast
→ all models predict hourly power 1h ahead
```

### Feature engineering

Every input row contains 16 features:

| Feature | Type | Description |
|---|---|---|
| `ghi` | Raw | Global Horizontal Irradiance (W/m²) |
| `poa_irradiance` | Raw | Plane-of-Array Irradiance (W/m²) |
| `t_ambient` | Raw | Ambient air temperature (°C) |
| `t_module` | Derived | Module temperature via NOCT model: `T_amb + (44−20)×GHI/800` |
| `wind_speed` | Raw | Wind speed (m/s) — affects module cooling |
| `hour_sin` | Derived | `sin(2π × hour / 24)` — cyclical hour encoding |
| `hour_cos` | Derived | `cos(2π × hour / 24)` — cyclical hour encoding |
| `doy_sin` | Derived | `sin(2π × doy / 365)` — cyclical day-of-year |
| `doy_cos` | Derived | `cos(2π × doy / 365)` — cyclical day-of-year |
| `lat` | Geographic | Latitude — continuous, enables unseen-location generalisation |
| `lon` | Geographic | Longitude — continuous, replaces one-hot site encoding |
| `power_lag_1` | Lag | Actual AC power output 1 hour ago |
| `power_lag_2` | Lag | Actual AC power output 2 hours ago |
| `power_lag_3` | Lag | Actual AC power output 3 hours ago |
| `ghi_lag_1` | Lag | GHI 1 hour ago — cloud persistence signal |
| `ghi_delta` | Derived | GHI change from last hour — indicates clearing or clouding |

**Why cyclical encoding?** Hour 23 and hour 0 are adjacent in time but numerically
far apart. `sin`/`cos` encoding wraps the values so the model sees that adjacency.

**Why lat/lon instead of one-hot site IDs?** One-hot encodings break for locations
not seen during training. Continuous coordinates let the model interpolate between
known sites and predict for any location on Earth.

**Why lag features?** For 1-hour ahead forecasting, recent power output and
irradiance trend carry strong predictive signal. A rising GHI over the past hour
suggests clearing skies; a falling one suggests incoming cloud cover.

### Data splits

**Time-based (default):** The last 20% of each site's time series is held out
for testing. This is the most realistic protocol as we should never train on the future.

---

## Repository Structure

```
solar-forecasting/
│
├── src/
│   ├── data_loader.py      ← loads NSRDB CSVs, computes AC power via physics model
│   ├── preprocessing.py    ← feature engineering, lag features, train/test splits
│   ├── models.py           ← all four model definitions (LR, LSTM, CNN1D, CNN-LSTM)
│   ├── train.py            ← end-to-end training script
│   └── evaluate.py         ← metrics, plots, evaluation
│
├── app/
│   └── app.py              ← Streamlit inference interface
│
├── data/                   ← place NSRDB CSV files here (gitignored)
│   ├── Phoenix_2017.csv ... Phoenix_2019.csv
│   ├── Denver_2017.csv  ... Denver_2019.csv
│   ├── Seattle_2017.csv ... Seattle_2019.csv
│   ├── Miami_2017.csv   ... Miami_2019.csv
│   └── Boston_2017.csv  ... Boston_2019.csv
│
├── results/
│   ├── linear_regression.pkl
│   ├── lstm_best.keras
│   ├── cnn1d_best.keras
│   ├── cnn_lstm_best.keras
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   ├── config.json
│   ├── metrics.json
│   ├── lstm_history.json
│   ├── cnn1d_history.json
│   ├── cnn_lstm_history.json
│   └── plots/
│       ├── actual_vs_predicted.png
│       ├── residuals.png
│       ├── model_comparison.png
│       ├── training_curves.png
│       └── time_series_comparison.png
│
├── .gitignore
├── requirements.txt
└── README.md
```
---

## Dataset

### Source

Real measured solar radiation data from the **NREL National Solar Radiation
Database (NSRDB)** — a satellite-derived, serially complete collection of
hourly irradiance and meteorological data for the United States.

Downloaded manually from [nsrdb.nlr.gov/data-viewer](https://nsrdb.nlr.gov/data-viewer).
No API key required for download.

### Sites

| # | Location | Climate | Lat | Lon |
|---|---|---|---|---|
| 1 | Phoenix, AZ | Hot Desert | 33.45°N | 112.07°W |
| 2 | Denver, CO | Cold Continental | 39.74°N | 104.98°W |
| 3 | Seattle, WA | Marine | 47.61°N | 122.33°W |
| 4 | Miami, FL | Humid Subtropical | 25.77°N | 80.19°W |
| 5 | Boston, MA | Temperate | 42.36°N | 71.06°W |

### Processing steps

1. Download hourly CSV files from NSRDB viewer (2017, 2018, 2019 per site)
2. Filter nighttime records where GHI < 50 W/m²
3. Compute POA irradiance and module temperature via NOCT model
4. Compute AC power output using efficiency-based PV model
5. Subsample to 6,000 rows per site (30,000 total, balanced across years)
6. Inject 2% missing values in temperature and wind columns (realistic)
7. Impute missing values using per-site column median
---

## Model Architectures

### 1. Linear Regression (Baseline)

Ordinary least-squares on a flattened 24-hour window (384 features = 24 × 16).
Trained on the same past-window information as the deep models for a fair
comparison. Fast to train but cannot capture nonlinear relationships or
sequential patterns.

```
Input (24 × 16 = 384,) → LinearRegression → Output: AC Power 1h ahead (W)
```

### 2. LSTM

Long Short-Term Memory network. Processes a 24-hour sliding window and uses
gated memory cells to capture temporal dependencies like cloud persistence
and thermal inertia, which gives it a genuine advantage over linear models for
forecasting tasks.

```
Input (24, 16)
    → LSTM(64 units)
    → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(1, linear)
    → Output: AC Power 1h ahead (W)
```

### 3. 1D-CNN

One-dimensional convolutional network. Uses learned filters to detect local
patterns in the 24-hour window (e.g. morning irradiance ramp-up, cloud
transients). Faster to train than LSTM but less effective at long-range
temporal dependencies.

```
Input (24, 16)
    → Conv1D(64 filters, kernel=3, ReLU)
    → MaxPool1D(2)
    → Conv1D(32 filters, kernel=3, ReLU)
    → GlobalAveragePooling1D
    → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(1, linear)
    → Output: AC Power 1h ahead (W)
```

### 4. CNN-LSTM Hybrid

The CNN block extracts local irradiance patterns from the window. The LSTM
then models how those features evolve across the full 24-hour sequence.
It combines the pattern-recognition strength of CNNs with the sequential memory
of LSTMs and is competitive with a standalone LSTM on this dataset.

```
Input (24, 16)
    → Conv1D(64 filters, kernel=3, ReLU)   ← local feature extraction
    → MaxPool1D(2)
    → Conv1D(32 filters, kernel=3, ReLU)   ← higher-order features
    → LSTM(64 units)                        ← temporal sequence modelling
    → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(1, linear)
    → Output: AC Power 1h ahead (W)
```

### Training configuration (all deep models)

| Parameter | Value |
|---|---|
| Loss function | Mean Squared Error (MSE) |
| Optimiser | Adam |
| Learning rate | 0.001 (with ReduceLROnPlateau) |
| Batch size | 32 |
| Max epochs | 80 |
| Early stopping patience | 12 epochs |
| Validation split | 10% of training data |
| Sliding window | 24 hours |
| Forecast horizon | 1 hour ahead |

---

## Quick Start — Local

### Prerequisites

- Python 3.10 or newer
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/alexs-44/solar-forecasting.git
cd solar-forecasting
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Conda users:** `conda create -n solar python=3.10 && conda activate solar`
> then run the pip command above.

### Step 3 — Download NSRDB data

Go to [nsrdb.nlr.gov/data-viewer](https://nsrdb.nlr.gov/data-viewer) and
download hourly CSV files for each of the 5 sites for years 2017, 2018,
and 2019 (15 files total). Select these attributes: GHI, DHI, DNI,
Temperature, Wind Speed, Solar Zenith Angle, Relative Humidity.
Enable "Convert UTC to Local Time". Save files to the `data/` folder
using this naming convention:

```
data/Phoenix_2017.csv, Phoenix_2018.csv, Phoenix_2019.csv
data/Denver_2017.csv  ... and so on for all 5 sites
```

### Step 4 — Build dataset

```bash
python src/data_loader.py
```

Processes all 15 CSVs, computes AC power via physics model, and saves
`data/nsrdb_combined.csv` (~30,000 rows).

### Step 5 — Train all models

```bash
HORIZON=1 python src/train.py
```

Trains Linear Regression, LSTM, 1D-CNN, and CNN-LSTM. Saves all model
files to `results/`. On CPU this takes roughly 45–60 minutes for the
deep models. On a GPU it takes 5–10 minutes total.

### Step 6 — Evaluate and generate plots

```bash
python src/evaluate.py
```

Prints a metrics table and saves all plots to `results/plots/`.

### Step 7 — Launch the app

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running the App

The Streamlit app requires trained model files in `results/`. After running
`train.py`, launch with:

```bash
streamlit run app/app.py
```

### How to use it

1. **Select a preset location** from the dropdown, or type a custom latitude/longitude
2. **Choose a forecast date** (up to 7 days ahead)
3. Click **Run Forecast**
4. The app fetches live hourly weather from [Open-Meteo](https://open-meteo.com)
   (free, no API key required) and displays:
   - Peak predicted power (kW) and daily energy yield (kWh) per model
   - 24-hour power forecast chart with GHI overlay
   - Raw weather data table
   - Hourly power breakdown table

### Weather variables used from Open-Meteo

| Open-Meteo variable | Used as |
|---|---|
| `shortwave_radiation` | GHI (W/m²) |
| `temperature_2m` | Ambient temperature (°C) |
| `windspeed_10m` | Wind speed (m/s) |

---

## Reproducing Results

All data and models are regenerated from scratch by the pipeline — no pre-trained
files need to be downloaded.

```bash
# Full reproduction from scratch
python src/data_loader.py   # ~1 min  — fetch real PVDAQ data
python src/train.py          # ~5 min — train all 4 models
python src/evaluate.py       # ~1 min  — metrics + plots
```

All random seeds are fixed for reproducibility:
- `numpy.random.default_rng(42)` for data processing
- `random_state=42` for scikit-learn
- TensorFlow global seed set at import in `train.py`

---

## Limitations 

Models were trained on 5 North American sites (25°N–48°N latitude and 71°N–122°W longitude).
Predictions for locations significantly outside this range, particularly
different climate zones or continents, may be unreliable due to
distribution shift.

------

## Weather API

The Streamlit app uses [Open-Meteo](https://open-meteo.com) for live weather
forecasts. It is completely free and requires no registration or API key.

---

## Configuration

Override training defaults with environment variables:

```bash
# Example: change forecast horizon, train for 100 epochs
HORIZON=3 LSTM_EPOCHS=100 python src/train.py
```

| Variable | Default | Description |
|---|---|---|
| `SPLIT_MODE` | `time` | `"time"` = 80/20 chronological, `"site"` = cross-site |
| `TEST_SITE` | last site ID | Site ID withheld for testing (site split only) |
| `LSTM_EPOCHS` | `80` | Maximum training epochs for all deep models |
| `LSTM_BATCH` | `32` | Batch size for all deep models |
| `HORIZON` | `1` | Hours ahead to forecast |

---

## Troubleshooting

**`TensorFlow not installed`**
→ Run `pip install tensorflow`. On Apple Silicon Macs: `pip install tensorflow-macos`.
→ The pipeline still runs Linear Regression without TensorFlow.

**`FileNotFoundError: Phoenix_2019.csv`**
→ Make sure all 15 NSRDB CSV files are in the `data/` folder with exact names
  like `Phoenix_2019.csv`, `Denver_2018.csv` etc.

**Streamlit app shows "No trained models found"**
→ Run `HORIZON=1 python src/train.py` before launching the app.

**Open-Meteo returns no data**
→ The forecast window is limited to 7 days ahead. Choose a date within that range.

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
tensorflow>=2.13
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.28
requests>=2.31
joblib>=1.3
scipy>=1.11
```

Install everything: `pip install -r requirements.txt`

---

## Author

**Alex Sol**
GitHub: [@alexs-44](https://github.com/alexs-44)

---

## Acknowledgements

- [NREL NSRDB](https://nsrdb.nlr.gov/data-viewer) — real measured solar radiation data
- [Open-Meteo](https://open-meteo.com) — free weather forecast API
- Iqbal (1983), *An Introduction to Solar Radiation* — clear-sky model reference

---

## License

MIT License — see `LICENSE` for details.
