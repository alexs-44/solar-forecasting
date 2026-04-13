"""
evaluate.py
===========
Evaluation script for the forecasting model (v3).

All models are evaluated on their ability to predict power HORIZON hours ahead
using only past weather data — not current conditions.
"""

from __future__ import annotations

import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models import load_sklearn_model, TF_AVAILABLE

if TF_AVAILABLE:
    from src.models import load_keras_model

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

PALETTE = {
    "Linear Regression": "#4C72B0",
    "LSTM":              "#55A868",
    "CNN1D":             "#DD8452",
    "CNN-LSTM":          "#C44E52",
}


def metrics(yt, yp):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE":  float(mean_absolute_error(yt, yp)),
        "R2":   float(r2_score(yt, yp)),
    }


def load_config():
    cfg_path = RESULTS_DIR / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {"horizon": 1, "window": 24}


def plot_actual_vs_predicted(models_data, horizon):
    n = len(models_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, yt, yp) in zip(axes, models_data):
        m   = metrics(yt, yp)
        lim = max(yt.max(), yp.max()) * 1.05
        ax.scatter(yt, yp, alpha=0.3, s=6, color=PALETTE.get(name, "#888"))
        ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="Ideal")
        ax.set_title(
            f"{name}\nRMSE={m['RMSE']:.1f}W  R²={m['R2']:.4f}",
            fontsize=11,
        )
        ax.set_xlabel(f"Actual power at t+{horizon}h (W)")
        ax.set_ylabel(f"Predicted power at t+{horizon}h (W)")
    plt.suptitle(
        f"Actual vs Predicted — {horizon}-Hour Ahead Forecast", fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → actual_vs_predicted.png")


def plot_residuals(models_data):
    n = len(models_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, yt, yp) in zip(axes, models_data):
        res = yp - yt
        ax.hist(res, bins=50, color=PALETTE.get(name, "#888"),
                alpha=0.75, edgecolor="white")
        ax.axvline(0, color="k", lw=1.5, ls="--")
        ax.axvline(res.mean(), color="red", lw=1.5,
                   label=f"Mean={res.mean():.0f}W")
        ax.set_title(f"{name} — Residuals")
        ax.set_xlabel("Residual (W)")
        ax.legend(fontsize=9)
    plt.suptitle("Residual Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → residuals.png")


def plot_model_comparison(results):
    names  = list(results.keys())
    colors = [PALETTE.get(n, "#888") for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(metric, fontsize=13)
        fmt = ".4f" if metric == "R2" else ".1f"
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=10)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Model Comparison — Forecasting Test Set", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → model_comparison.png")


def plot_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (fname, label) in zip(axes,
            [("lstm", "LSTM"), ("cnn1d", "CNN1D"), ("cnn_lstm", "CNN-LSTM")]):
        path = RESULTS_DIR / f"{fname}_history.json"
        if not path.exists():
            ax.text(0.5, 0.5, f"{label}\nnot trained",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        with open(path) as f:
            h = json.load(f)
        epochs = range(1, len(h["loss"]) + 1)
        ax.plot(epochs, h["loss"],     label="Train", color=PALETTE.get(label, "#888"))
        ax.plot(epochs, h["val_loss"], label="Val",
                color=PALETTE.get(label, "#888"), ls="--", alpha=0.7)
        ax.set_title(f"{label} — Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
    plt.suptitle("Deep Learning Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_curves.png", dpi=150)
    plt.close()
    print("  Saved → training_curves.png")


def plot_time_series(models_data, horizon, n_samples=200):
    fig, ax = plt.subplots(figsize=(16, 5))
    yt = models_data[0][1]
    ax.plot(yt[:n_samples], label=f"Actual (t+{horizon}h)", color="black", lw=2)
    styles = ["--", "-", "-.", ":"]
    for (name, _, yp), ls in zip(models_data, styles):
        ax.plot(yp[:n_samples], label=name,
                color=PALETTE.get(name, "#888"), lw=1.2, ls=ls)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("AC Power (W)")
    ax.set_title(f"{horizon}-Hour Ahead Forecast — All Models vs Actual",
                 fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "time_series_comparison.png", dpi=150)
    plt.close()
    print("  Saved → time_series_comparison.png")


def evaluate_all():
    cfg     = load_config()
    horizon = cfg["horizon"]

    print("\n" + "=" * 60)
    print(f"  Evaluation  |  Forecast horizon = {horizon}h ahead")
    print("=" * 60)

    # Load flat sequences for Linear Regression
    X_flat  = np.load(RESULTS_DIR / "X_flat_test.npy")
    y_flat  = np.load(RESULTS_DIR / "y_flat_test.npy")

    # Load sequential data for deep models
    X_seq = y_seq = None
    if (RESULTS_DIR / "X_seq_test.npy").exists():
        X_seq = np.load(RESULTS_DIR / "X_seq_test.npy")
        y_seq = np.load(RESULTS_DIR / "y_seq_test.npy")

    results     = {}
    models_data = []

    # ── Linear Regression ─────────────────────────────────────────────────────
    print("\n  Linear Regression")
    lr    = load_sklearn_model("linear_regression")
    yp_lr = np.clip(lr.predict(X_flat), 0, None)
    results["Linear Regression"] = metrics(y_flat, yp_lr)
    models_data.append(("Linear Regression", y_flat, yp_lr))

    # ── Deep models ───────────────────────────────────────────────────────────
    if TF_AVAILABLE and X_seq is not None:
        for key, label in [("lstm","LSTM"), ("cnn1d","CNN1D"),
                            ("cnn_lstm","CNN-LSTM")]:
            ckpt = RESULTS_DIR / f"{key}_best.keras"
            if not ckpt.exists():
                print(f"  [SKIP] {label} — checkpoint not found")
                continue
            print(f"\n  {label}")
            model = load_keras_model(key)
            yp    = np.clip(
                model.predict(X_seq, batch_size=64, verbose=0).flatten(), 0, None
            )
            results[label] = metrics(y_seq, yp)
            models_data.append((label, y_seq, yp))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_actual_vs_predicted(models_data, horizon)
    plot_residuals(models_data)
    plot_model_comparison(results)
    plot_training_curves()
    plot_time_series(models_data, horizon)

    # ── Metrics table ──────────────────────────────────────────────────────────
    print("\n" + "─" * 58)
    print(f"  {'Model':<22} {'RMSE (W)':>10} {'MAE (W)':>10} {'R²':>8}")
    print("  " + "─" * 54)
    for name, m in results.items():
        print(f"  {name:<22} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} {m['R2']:>8.4f}")
    print("─" * 58)
    print(f"\n  Forecast horizon: {horizon} hour(s) ahead")

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics → results/metrics.json")
    print(f"  Plots   → results/plots/")


if __name__ == "__main__":
    evaluate_all()
