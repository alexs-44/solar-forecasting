"""
train.py
========
Code writing assisted by Claude AI
========
End-to-end training script — SOLAR FORECASTING MODEL. 

  - Linear Regression trains on flattened 24h windows (same info as deep models)
  - All models predict power HORIZON hours ahead, not current power
  - This makes the comparison fair and scientifically meaningful

Usage:
    python src/train.py

Environment variable overrides:
    SPLIT_MODE    = "time" | "site"   (default: time)
    TEST_SITE     = int               (default: last site)
    LSTM_EPOCHS   = int               (default: 80)
    LSTM_BATCH    = int               (default: 32)
    HORIZON       = int               (default: 1, hours ahead to predict)
"""

from __future__ import annotations

import os, sys, time, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    load_and_preprocess, make_sequences, make_flat_sequences, DEFAULT_HORIZON
)
from src.models import (
    build_linear, build_lstm, build_cnn1d, build_cnn_lstm,
    get_callbacks, save_sklearn_model, TF_AVAILABLE,
)

SPLIT_MODE  = os.getenv("SPLIT_MODE",   "time")
TEST_SITE   = os.getenv("TEST_SITE",    None)
EPOCHS      = int(os.getenv("LSTM_EPOCHS", 80))
BATCH       = int(os.getenv("LSTM_BATCH",  32))
HORIZON     = int(os.getenv("HORIZON", DEFAULT_HORIZON))
WINDOW      = 24
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def train_deep_model(model, name, X_seq_tr, y_seq_tr):
    banner(f"Training {name}")
    t0 = time.time()
    history = model.fit(
        X_seq_tr, y_seq_tr,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=get_callbacks(name, patience=12),
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  |  {len(history.history['loss'])} epochs")

    hist_data = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    with open(RESULTS_DIR / f"{name.lower()}_history.json", "w") as f:
        json.dump(hist_data, f, indent=2)
    return history


def train_all():
    data_path = Path("data/pvdaq_combined.csv")
    if not data_path.exists():
        print("\n  No dataset found. Run python src/data_loader.py first.")
        sys.exit(1)

    # ── Preprocess ────────────────────────────────────────────────────────────
    banner(f"Preprocessing  |  Forecast horizon = {HORIZON}h ahead")
    test_site = int(TEST_SITE) if TEST_SITE else None
    (X_train, y_train, X_test, y_test,
     scaler, feature_cols,
     df_train, df_test) = load_and_preprocess(
        data_path=data_path,
        split=SPLIT_MODE,
        test_site_id=test_site,
        save_scaler=True,
    )

    sid_tr = df_train["site_id"].values
    sid_te = df_test["site_id"].values
    n_features = X_train.shape[1]

    # ── Build sequences ───────────────────────────────────────────────────────
    banner("Building Sliding-Window Sequences")
    print(f"  Window = {WINDOW}h  |  Horizon = {HORIZON}h ahead")

    # Sequences for deep models: shape (n, window, features)
    X_seq_tr, y_seq_tr = make_sequences(X_train, y_train, WINDOW, HORIZON, sid_tr)
    X_seq_te, y_seq_te = make_sequences(X_test,  y_test,  WINDOW, HORIZON, sid_te)
    print(f"  Deep model sequences — train: {X_seq_tr.shape}, test: {X_seq_te.shape}")

    # Flat sequences for Linear Regression: shape (n, window * features)
    # LR now sees the same 24h of past data as LSTM/CNN — fair comparison
    X_flat_tr, y_flat_tr = make_flat_sequences(X_train, y_train, WINDOW, HORIZON, sid_tr)
    X_flat_te, y_flat_te = make_flat_sequences(X_test,  y_test,  WINDOW, HORIZON, sid_te)
    print(f"  Linear Regression flat — train: {X_flat_tr.shape}, test: {X_flat_te.shape}")

    # Save test sequences
    np.save(RESULTS_DIR / "X_seq_test.npy",  X_seq_te)
    np.save(RESULTS_DIR / "y_seq_test.npy",  y_seq_te)
    np.save(RESULTS_DIR / "X_flat_test.npy", X_flat_te)
    np.save(RESULTS_DIR / "y_flat_test.npy", y_flat_te)

    # Save horizon for evaluate.py to display
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump({"horizon": HORIZON, "window": WINDOW}, f)

    # ── 1. Linear Regression — trained on flattened 24h window ───────────────
    banner("Training Linear Regression  (flattened 24h window input)")
    t0 = time.time()
    lr = build_linear()
    lr.fit(X_flat_tr, y_flat_tr)
    print(f"  Done in {time.time()-t0:.1f}s")
    save_sklearn_model(lr, "linear_regression")

    # ── 2–4. Deep models ──────────────────────────────────────────────────────
    if TF_AVAILABLE:
        lstm     = build_lstm(n_features, WINDOW)
        lstm.summary()
        train_deep_model(lstm, "lstm", X_seq_tr, y_seq_tr)

        cnn      = build_cnn1d(n_features, WINDOW)
        cnn.summary()
        train_deep_model(cnn, "cnn1d", X_seq_tr, y_seq_tr)

        cnn_lstm = build_cnn_lstm(n_features, WINDOW)
        cnn_lstm.summary()
        train_deep_model(cnn_lstm, "cnn_lstm", X_seq_tr, y_seq_tr)
    else:
        print("\n  [SKIP] Deep learning — install TensorFlow to train LSTM/CNN")

    # Save raw test data for reference
    np.save(RESULTS_DIR / "X_test.npy",  X_test)
    np.save(RESULTS_DIR / "y_test.npy",  y_test)
    df_test.to_csv(RESULTS_DIR  / "df_test.csv",  index=False)
    df_train.to_csv(RESULTS_DIR / "df_train.csv", index=False)

    banner(f"Training Complete — horizon={HORIZON}h  |  artefacts in results/")
    print("  Next: python src/evaluate.py")


if __name__ == "__main__":
    train_all()
