"""
models.py
========
Code writing assisted by Claude AI
=========
Model definitions for all 4 architectures:

  1. LinearRegression      — scikit-learn baseline, flat input
  2. LSTM                  — 24h sliding window, sequential memory
  3. CNN1D                 — 24h sliding window, local pattern detection via convolution
  4. CNN_LSTM              — 24h sliding window, CNN feature extractor → LSTM temporal model

Architecture rationale
----------------------
All three deep learning models share the same 24-hour sliding window input so
their results are directly comparable. The CNN acts as a local feature extractor
(detecting intra-day irradiance patterns) before the LSTM captures longer-range
temporal dependencies in the hybrid model.

        Input (24h, F features)
               │
        ┌──────┴──────────────────────┐
        │ LSTM          │ CNN1D       │ CNN-LSTM          │
        │ ─────         │ ─────       │ ──────────        │
        │ LSTM(64)      │ Conv1D(64)  │ Conv1D(64)→MaxPool│
        │ Dropout(0.1)  │ MaxPool     │ Conv1D(32)→MaxPool│
        │ Dense(32,ReLU)│ Flatten     │ LSTM(64)          │
        │               │ Dense(32)   │ Dropout(0.1)      │
        │               │             │ Dense(32,ReLU)    │
        └──────┬──────────────────────┘
               │
          Dense(1, linear) → AC Power (W)
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not installed — deep learning models unavailable.")

RESULTS_DIR = Path("results")


# ─────────────────────────────────────────────────
# 1. Linear Regression (baseline)
# ─────────────────────────────────────────────────

def build_linear() -> LinearRegression:
    """Ordinary least-squares baseline. Flat (non-sequential) input."""
    return LinearRegression()


# ─────────────────────────────────────────────────
# 2. LSTM
# ─────────────────────────────────────────────────

def build_lstm(
    n_features: int,
    window: int = 24,
    lstm_units: int = 64,
    dropout: float = 0.1,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """
    Single-layer LSTM for sequential solar forecasting.

    Input  : (batch, window=24, n_features)
    LSTM   : lstm_units units
    Dropout: dropout rate
    Dense  : dense_units units, ReLU
    Output : 1 unit, linear
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for LSTM.")

    inputs = keras.Input(shape=(window, n_features), name="input")
    x = keras.layers.LSTM(lstm_units, name="lstm")(inputs)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    out = keras.layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs, out, name="LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse", metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────
# 3. 1D-CNN
# ─────────────────────────────────────────────────

def build_cnn1d(
    n_features: int,
    window: int = 24,
    filters: int = 64,
    kernel_size: int = 3,
    dense_units: int = 32,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """
    1D Convolutional network for local pattern detection in the 24h window.

    Conv1D scans the time axis with learned filters, detecting local irradiance
    shapes (e.g. morning ramp-up, cloud transients) regardless of position.
    Faster to train than LSTM, good at recognising recurring daily patterns.

    Input      : (batch, window=24, n_features)
    Conv1D(64) : kernel_size=3, ReLU — local feature maps
    MaxPool    : stride 2 — downsample
    Conv1D(32) : kernel_size=3, ReLU — higher-level features
    GlobalAvgPool → Dropout → Dense(32, ReLU) → Output
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for CNN1D.")

    inputs = keras.Input(shape=(window, n_features), name="input")
    x = keras.layers.Conv1D(filters, kernel_size, activation="relu",
                             padding="same", name="conv1")(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = keras.layers.Conv1D(filters // 2, kernel_size, activation="relu",
                             padding="same", name="conv2")(x)
    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    out = keras.layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs, out, name="CNN1D")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse", metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────
# 4. CNN-LSTM Hybrid
# ─────────────────────────────────────────────────

def build_cnn_lstm(
    n_features: int,
    window: int = 24,
    cnn_filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 64,
    dropout: float = 0.1,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """
    CNN-LSTM hybrid: convolutional feature extraction followed by LSTM
    temporal modelling.

    The CNN layers learn to compress each local time window into a rich feature
    vector; the LSTM then models how those features evolve over the full 24h
    sequence. This combines the pattern-recognition strength of CNNs with the
    sequential memory of LSTMs.

    Input          : (batch, window=24, n_features)
    Conv1D(64)     : local feature extraction
    MaxPool        : downsample
    Conv1D(32)     : higher-order features
    LSTM(64)       : temporal sequence modelling
    Dropout(0.1)
    Dense(32, ReLU)
    Output(1, linear)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for CNN-LSTM.")

    inputs = keras.Input(shape=(window, n_features), name="input")

    # CNN block
    x = keras.layers.Conv1D(cnn_filters, kernel_size, activation="relu",
                             padding="same", name="conv1")(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = keras.layers.Conv1D(cnn_filters // 2, kernel_size, activation="relu",
                             padding="same", name="conv2")(x)

    # LSTM block (return_sequences=False — we only need the final state)
    x = keras.layers.LSTM(lstm_units, name="lstm")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    out = keras.layers.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(inputs, out, name="CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse", metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────
# Shared training callbacks
# ─────────────────────────────────────────────────

def get_callbacks(model_name: str, patience: int = 12) -> list:
    """
    Standard callbacks for all deep learning models:
      - EarlyStopping      : stop when val_loss stops improving
      - ModelCheckpoint    : save best weights
      - ReduceLROnPlateau  : halve LR after 5 stagnant epochs
    """
    if not TF_AVAILABLE:
        return []

    ckpt_path = f"results/{model_name.lower()}_best.keras"
    Path("results").mkdir(exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor="val_loss",
            save_best_only=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1,
        ),
    ]


# ─────────────────────────────────────────────────
# Persistence helpers for saving and loading trained models to and from disk.
# ─────────────────────────────────────────────────

def save_sklearn_model(model, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  Saved → {path}")
    return path


def load_sklearn_model(name: str):
    return joblib.load(RESULTS_DIR / f"{name}.pkl")


def load_keras_model(name: str):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required.")
    return keras.models.load_model(f"results/{name}_best.keras")
