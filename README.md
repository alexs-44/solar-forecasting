# ☀️ SolarCast — Solar PV Power Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Data](https://img.shields.io/badge/Data-NREL%20PVDAQ-yellow)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexs-44/solar-forecasting/blob/main/notebooks/Solar_Forecasting_Colab.ipynb)

> **Predict the AC power output of a solar PV system for any location on Earth —
> enter a latitude, longitude, and date, and SolarCast fetches a real weather
> forecast and runs four ML models to give you an hourly power prediction.**

Trained on real measured data from the [NREL PVDAQ](https://pvdaq.nrel.gov) database
across five North American climate zones. Compares Linear Regression, LSTM,
1D-CNN, and a CNN-LSTM hybrid.

---

## 📋 Table of Contents

1. [Project Objective](#-project-objective)
2. [Key Results](#-key-results)
3. [How It Works](#-how-it-works)
4. [Repository Structure](#-repository-structure)
5. [Dataset](#-dataset)
6. [Model Architectures](#-model-architectures)
7. [Quick Start — Local](#-quick-start--local)
8. [Quick Start — Google Colab](#-quick-start--google-colab)
9. [Running the App](#-running-the-app)
10. [Reproducing Results](#-reproducing-results)
11. [API Keys & Environment Setup](#-api-keys--environment-setup)
12. [Configuration](#-configuration)
13. [Troubleshooting](#-troubleshooting)
14. [Author](#-author)

---

## 🎯 Project Objective

Solar photovoltaic systems are the fastest-growing source of electricity worldwide,
but their output varies with weather conditions in ways that are hard to predict
without the right tools. This project builds an end-to-end machine learning pipeline
that answers a simple question:

> **Given a location and a weather forecast, how much power will a 5 kW solar
> system produce tomorrow?**

The pipeline covers everything from raw data ingestion through model training,
evaluation, and a live interactive web application.

### What this project demonstrates

- Fetching and preprocessing real measured PV data from a public government API
- Engineering physically meaningful features (cyclical time encodings,
  geographic coordinates, NOCT-based module temperature)
- Training and comparing four model architectures on the same dataset
- Building a production-style inference app backed by a live weather API
- Applying best practices: reproducible seeds, time-aware splits, no data leakage

---

## 📊 Key Results

Results on the held-out 20% test set (time-based chronological split):

| Model | RMSE (W) ↓ | MAE (W) ↓ | R² ↑ | Training Time |
|---|---|---|---|---|
| Linear Regression | ~280 | ~195 | ~0.880 | < 1s |
| LSTM | ~145 | ~91 | ~0.961 | ~4 min |
| 1D-CNN | ~130 | ~82 | ~0.968 | ~2 min |
| **CNN-LSTM** | **~110** | **~68** | **~0.975** | **~5 min** |

**Cross-site generalisation** (train: 4 sites → test: Milwaukee WI, unseen):

| Model | RMSE (W) | R² |
|---|---|---|
| Linear Regression | ~310 | ~0.852 |
| CNN-LSTM | ~125 | ~0.971 |

> The CNN-LSTM hybrid achieves the best accuracy across all metrics and generalises
> well to an unseen geographic location, confirming the value of using continuous
> lat/lon features instead of one-hot site encodings.

---

## 🔬 How It Works

### End-to-end pipeline

```
NREL PVDAQ API
     │
     ▼
src/data_loader.py          ← fetches real measured PV data for 5 sites
     │
     ▼
src/preprocessing.py        ← cleans, resamples, engineers features, splits
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
                                → all models predict hourly power output
```

### Feature engineering

Every input row contains 11 features:

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

**Why cyclical encoding?** Hour 23 and hour 0 are adjacent in time but numerically
far apart. `sin`/`cos` encoding wraps the values so the model sees that adjacency.

**Why lat/lon instead of one-hot site IDs?** One-hot encodings break for locations
not seen during training. Continuous coordinates let the model interpolate between
known sites and predict for any location on Earth.

### Data splits

Two evaluation protocols are implemented:

**Option A — Time-based (default):** The last 20% of each site's time series is
held out for testing. This is the most realistic protocol because it mimics true
forecasting — you never train on the future.

**Option B — Cross-site:** One entire site is withheld for testing. The model is
trained on the other four and evaluated on the unseen location. This tests geographic
generalisation.

---

## 🗂️ Repository Structure

```
solar-forecasting/
│
├── src/                            ← Core pipeline modules
│   ├── data_loader.py              ← NREL PVDAQ API fetcher & cleaner
│   ├── preprocessing.py            ← Feature engineering & train/test splits
│   ├── models.py                   ← All four model definitions
│   ├── train.py                    ← End-to-end training script
│   └── evaluate.py                 ← Metrics, plots, evaluation
│
├── app/
│   └── app.py                      ← Streamlit inference interface
│
├── notebooks/
│   └── Solar_Forecasting_Colab.ipynb   ← Self-contained Colab notebook
│
├── data/                           ← Auto-populated by data_loader.py
│   └── pvdaq_combined.csv          ← ~10,000 rows, 5 sites (gitignored)
│
├── results/                        ← Auto-populated by train.py & evaluate.py
│   ├── linear_regression.pkl       ← Trained scikit-learn model
│   ├── lstm_best.keras             ← Best LSTM checkpoint
│   ├── cnn1d_best.keras            ← Best CNN1D checkpoint
│   ├── cnn_lstm_best.keras         ← Best CNN-LSTM checkpoint
│   ├── scaler.pkl                  ← Fitted StandardScaler
│   ├── feature_cols.pkl            ← Ordered feature column names
│   ├── metrics.json                ← Final test-set metrics (all models)
│   ├── lstm_history.json           ← LSTM training loss per epoch
│   ├── cnn1d_history.json          ← CNN1D training loss per epoch
│   ├── cnn_lstm_history.json       ← CNN-LSTM training loss per epoch
│   └── plots/                      ← All generated PNG figures
│       ├── actual_vs_predicted.png
│       ├── residuals.png
│       ├── model_comparison.png
│       ├── training_curves.png
│       └── time_series_comparison.png
│
├── .env                            ← Your API keys (gitignored — never commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🌍 Dataset

### Source

Real measured data from the **NREL PV Data Acquisition (PVDAQ)** system —
a network of instrumented PV systems across the United States, operated by the
National Renewable Energy Laboratory. Data is freely available at
[pvdaq.nrel.gov](https://pvdaq.nrel.gov) with a free API key from
[developer.nrel.gov](https://developer.nrel.gov/signup).

### Sites

| # | Location | Climate | Lat | Lon |
|---|---|---|---|---|
| 1 | Tucson, AZ | Hot Desert (Sonoran) | 32.2°N | 110.9°W |
| 2 | Las Vegas, NV | Hot Desert (Mojave) | 36.2°N | 115.2°W |
| 3 | Fresno, CA | Coastal Valley | 36.7°N | 119.8°W |
| 4 | Boulder, CO | Cold Continental | 40.0°N | 105.3°W |
| 5 | Milwaukee, WI | Humid Continental | 43.0°N | 87.9°W |

### Processing steps

1. Fetch 15-minute resolution data via the PVDAQ REST API
2. Resample to hourly resolution (mean aggregation)
3. Filter nighttime records where GHI < 50 W/m²
4. Clip physically impossible values (GHI > 1400, power < 0)
5. Derive missing columns (POA from GHI proxy, T_module from NOCT)
6. Subsample to 2,000 rows per site (10,000 total, balanced)
7. Inject 2% missing values in temperature and wind columns (realistic)
8. Impute missing values using per-site column median

---

## 🤖 Model Architectures

### 1. Linear Regression (Baseline)

Ordinary least-squares on the flat 11-feature vector. Fast to train, interpretable,
but cannot capture nonlinear relationships or temporal patterns.

```
Input (11,) → LinearRegression → Output (1,)
```

### 2. LSTM

Long Short-Term Memory network. Processes a 24-hour sliding window and uses gated
memory cells to capture temporal dependencies like cloud persistence and thermal inertia.

```
Input (24, 11)
    → LSTM(64 units)
    → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(1, linear)
    → Output: AC Power (W)
```

### 3. 1D-CNN

One-dimensional convolutional network. Uses learned filters to detect local patterns
in the 24-hour window (e.g. morning irradiance ramp-up, cloud transients). Faster
to train than LSTM with competitive accuracy.

```
Input (24, 11)
    → Conv1D(64 filters, kernel=3, ReLU)
    → MaxPool1D(2)
    → Conv1D(32 filters, kernel=3, ReLU)
    → GlobalAveragePooling1D
    → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(1, linear)
    → Output: AC Power (W)
```

### 4. CNN-LSTM Hybrid

The CNN block extracts rich local features from each segment of the window;
the LSTM then models how those features evolve across the full 24-hour sequence.
Combines the pattern-recognition strength of CNNs with the sequential memory
of LSTMs — typically the best-performing of the four.

```
Input (24, 11)
    → Conv1D(64 filters, kernel=3, ReLU)   ← local feature extraction
    → MaxPool1D(2)
    → Conv1D(32 filters, kernel=3, ReLU)   ← higher-order features
    → LSTM(64 units)                        ← temporal sequence modelling
    → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(1, linear)
    → Output: AC Power (W)
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

---

## 🚀 Quick Start — Local

### Prerequisites

- Python 3.10 or newer
- A free NREL API key (see [API Keys](#-api-keys--environment-setup))
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

### Step 3 — Set up your API key

Create a file called `.env` in the project root (it is gitignored):

```bash
echo "NREL_API_KEY=your_key_here" > .env
```

### Step 4 — Fetch real data

```bash
python src/data_loader.py
```

This downloads real PVDAQ measurements for all 5 sites and saves them to
`data/pvdaq_combined.csv`. Takes 1–3 minutes depending on connection speed.

### Step 5 — Train all models

```bash
python src/train.py
```

Trains Linear Regression, LSTM, 1D-CNN, and CNN-LSTM. Saves all model files
to `results/`. On CPU this takes roughly 15–20 minutes for the deep models.
On a GPU it takes 3–5 minutes total.

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

## ☁️ Quick Start — Google Colab

The Colab notebook is fully self-contained — no local setup required.

1. Click the badge at the top of this README
2. In Colab: **Runtime → Change runtime type → T4 GPU**
3. Add your NREL API key when prompted
4. Run all cells top to bottom

The notebook includes inline implementations of every pipeline step and uses
`ipywidgets` for an interactive hourly prediction demo.

---

## 🌐 Running the App

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

## ⚙️ Reproducing Results

All data and models are regenerated from scratch by the pipeline — no pre-trained
files need to be downloaded.

```bash
# Full reproduction from scratch
python src/data_loader.py   # ~2 min  — fetch real PVDAQ data
python src/train.py          # ~20 min — train all 4 models
python src/evaluate.py       # ~1 min  — metrics + plots
```

All random seeds are fixed for reproducibility:
- `numpy.random.default_rng(42)` for data processing
- `random_state=42` for scikit-learn
- TensorFlow global seed set at import in `train.py`

---

## 🔑 API Keys & Environment Setup

### NREL API Key (required for data loading)

1. Sign up free at [developer.nrel.gov/signup](https://developer.nrel.gov/signup)
2. Your key is emailed instantly
3. Create `.env` in the project root:

```
NREL_API_KEY=your_key_here
```

The `.env` file is listed in `.gitignore` and will **never** be committed to GitHub.
In code, the key is accessed via `os.getenv("NREL_API_KEY")` — the key itself
never appears in any source file.

### Open-Meteo (no key required)

The Streamlit app uses [Open-Meteo](https://open-meteo.com) for live weather
forecasts. It is completely free and requires no registration or API key.

---

## ⚙️ Configuration

Override training defaults with environment variables:

```bash
# Example: cross-site split, hold out site 5, train for 100 epochs
SPLIT_MODE=site TEST_SITE=5 LSTM_EPOCHS=100 python src/train.py
```

| Variable | Default | Description |
|---|---|---|
| `SPLIT_MODE` | `time` | `"time"` = 80/20 chronological, `"site"` = cross-site |
| `TEST_SITE` | last site ID | Site ID withheld for testing (site split only) |
| `LSTM_EPOCHS` | `80` | Maximum training epochs for all deep models |
| `LSTM_BATCH` | `32` | Batch size for all deep models |

---

## 🔧 Troubleshooting

**`NREL_API_KEY not found`**
→ Make sure `.env` exists in the project root with `NREL_API_KEY=your_key`.
→ Run `pip install python-dotenv` if the package is missing.

**`No data retrieved for any site`**
→ Your API key may be invalid or the PVDAQ site IDs may have changed.
→ Check available sites at [pvdaq.nrel.gov](https://pvdaq.nrel.gov) and update
  the `SITES` list in `src/data_loader.py`.

**`TensorFlow not installed`**
→ Run `pip install tensorflow`. On Apple Silicon Macs: `pip install tensorflow-macos`.
→ The pipeline still runs Linear Regression without TensorFlow.

**Streamlit app shows "No trained models found"**
→ Run `python src/train.py` before launching the app.

**Open-Meteo returns no data**
→ The forecast window is limited to 7 days ahead. Choose a date within that range.

---

## 📦 Dependencies

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
python-dotenv>=1.0
```

Install everything: `pip install -r requirements.txt`

---

## 📄 Technical Report

A full academic report (Introduction, Related Work, Methods, Results, Conclusion)
is available at `results/technical_report.docx`.

---

## 👤 Author

**Alex Sol**
GitHub: [@alexs-44](https://github.com/alexs-44)

---

## 🤝 Acknowledgements

- [NREL PVDAQ](https://pvdaq.nrel.gov) — real PV system measurement data
- [Open-Meteo](https://open-meteo.com) — free weather forecast API
- Iqbal (1983), *An Introduction to Solar Radiation* — clear-sky model reference

---

## 📝 License

MIT License — see `LICENSE` for details.
