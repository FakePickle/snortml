#!/usr/bin/env python3
"""
recalibrate_threshold.py
=========================
Scores all benign flow CSVs using the current model artifacts
and sets the threshold at the target FPR percentile.

Usage:
    python recalibrate_threshold.py \
        --models-dir D:/Programming/snortml_inference_final/models \
        --benign-dir D:/flows_csv \
        --target-fpr 0.01

Overwrites threshold.json and norm_stats.json in-place.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as rt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_artifacts(models_dir: Path):
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    onnx_path = models_dir / "snortml_autoencoder_int8.onnx"
    if not onnx_path.exists():
        onnx_path = models_dir / "snortml_autoencoder_fp32.onnx"
        logger.warning("INT8 model not found, using FP32")

    session = rt.InferenceSession(str(onnx_path), opts)
    input_name = session.get_inputs()[0].name

    scaler = joblib.load(models_dir / "scaler.joblib")

    with open(models_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)

    with open(models_dir / "preprocessor_meta.json") as f:
        meta = json.load(f)
    fill_values = meta.get("fill_values", {})
    clip_values = {k: tuple(v) for k, v in meta.get("clip_values", {}).items()}

    with open(models_dir / "threshold.json") as f:
        td = json.load(f)
    mse_weight = td.get("mse_weight", 0.6)
    mahal_weight = td.get("mahal_weight", 0.4)

    mahal_mean = np.load(str(models_dir / "mahal_mean.npy"))
    mahal_cov_inv = np.load(str(models_dir / "mahal_cov_inv.npy"))

    with open(models_dir / "norm_stats.json") as f:
        norm_stats = json.load(f)

    logger.info(f"Loaded model: {onnx_path.name} | features={len(feature_cols)}")
    return (
        session,
        input_name,
        scaler,
        feature_cols,
        fill_values,
        clip_values,
        mse_weight,
        mahal_weight,
        mahal_mean,
        mahal_cov_inv,
        norm_stats,
        td,
    )


def preprocess_csv(csv_path: Path, feature_cols, fill_values, clip_values, scaler):
    import pandas as pd

    logger.info(f"  Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    rows = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in df.columns:
            vals = df[col].fillna(fill_values.get(col, 0.0)).values.astype(np.float32)
        else:
            vals = np.full(len(df), fill_values.get(col, 0.0), dtype=np.float32)

        if col in clip_values:
            lo, hi = clip_values[col]
            vals = np.clip(vals, lo, hi)

        vals = np.where(np.isinf(vals), fill_values.get(col, 0.0), vals)
        rows[:, i] = vals

    X = scaler.transform(rows).astype(np.float32)
    return X


def score_batch(
    X,
    session,
    input_name,
    mahal_mean,
    mahal_cov_inv,
    norm_stats,
    mse_weight,
    mahal_weight,
):
    X_recon = session.run(None, {input_name: X})[0]
    err = X - X_recon

    mse = np.mean(err**2, axis=1)

    diff = err - mahal_mean
    mahal = np.einsum("ij,jk,ik->i", diff, mahal_cov_inv, diff)
    mahal = np.sqrt(np.maximum(mahal, 0))

    eps = 1e-8
    mse_norm = (mse - norm_stats["mse_lo"]) / max(
        norm_stats["mse_hi"] - norm_stats["mse_lo"], eps
    )
    mahal_norm = (mahal - norm_stats["mahal_lo"]) / max(
        norm_stats["mahal_hi"] - norm_stats["mahal_lo"], eps
    )

    scores = mse_weight * mse_norm + mahal_weight * mahal_norm
    return scores.astype(np.float32), mse, mahal


def main():
    parser = argparse.ArgumentParser(
        description="Recalibrate SnortML+ threshold on live benign traffic"
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--benign-dir", type=Path, required=True)
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Target false positive rate (default: 0.01 = 1%%)",
    )
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    (
        session,
        input_name,
        scaler,
        feature_cols,
        fill_values,
        clip_values,
        mse_weight,
        mahal_weight,
        mahal_mean,
        mahal_cov_inv,
        norm_stats,
        td,
    ) = load_artifacts(args.models_dir)

    csv_files = sorted(args.benign_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {args.benign_dir}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} benign CSV files")

    all_scores = []
    all_mse = []
    all_mahal = []

    for csv_path in csv_files:
        try:
            X = preprocess_csv(csv_path, feature_cols, fill_values, clip_values, scaler)
            logger.info(f"    {len(X):,} flows — scoring ...")

            for start in range(0, len(X), args.batch_size):
                batch = X[start : start + args.batch_size]
                scores, mse, mahal = score_batch(
                    batch,
                    session,
                    input_name,
                    mahal_mean,
                    mahal_cov_inv,
                    norm_stats,
                    mse_weight,
                    mahal_weight,
                )
                all_scores.append(scores)
                all_mse.append(mse)
                all_mahal.append(mahal)

        except Exception as e:
            logger.error(f"    Error on {csv_path.name}: {e}")
            continue

    if not all_scores:
        logger.error("No scores collected — check CSV paths and feature alignment")
        sys.exit(1)

    all_scores = np.concatenate(all_scores)
    all_mse = np.concatenate(all_mse)
    all_mahal = np.concatenate(all_mahal)

    logger.info(f"\nScore distribution over {len(all_scores):,} benign flows:")
    logger.info(
        f"  min={all_scores.min():.4f}  p50={np.percentile(all_scores, 50):.4f}"
        f"  p95={np.percentile(all_scores, 95):.4f}"
        f"  p99={np.percentile(all_scores, 99):.4f}"
        f"  p99.5={np.percentile(all_scores, 99.5):.4f}"
        f"  max={all_scores.max():.4f}"
    )

    # New norm stats from live benign distribution
    new_norm_stats = {
        "mse_lo": float(np.percentile(all_mse, 0.1)),
        "mse_hi": float(np.percentile(all_mse, 99.9)),
        "mahal_lo": float(np.percentile(all_mahal, 0.1)),
        "mahal_hi": float(np.percentile(all_mahal, 99.9)),
    }

    # Re-score with new norm stats
    eps = 1e-8
    mse_norm = (all_mse - new_norm_stats["mse_lo"]) / max(
        new_norm_stats["mse_hi"] - new_norm_stats["mse_lo"], eps
    )
    mahal_norm = (all_mahal - new_norm_stats["mahal_lo"]) / max(
        new_norm_stats["mahal_hi"] - new_norm_stats["mahal_lo"], eps
    )
    rescored = mse_weight * mse_norm + mahal_weight * mahal_norm

    target_percentile = (1.0 - args.target_fpr) * 100
    new_threshold = float(np.percentile(rescored, target_percentile))

    logger.info(f"\nNew norm stats: {new_norm_stats}")
    logger.info(
        f"New threshold at FPR={args.target_fpr * 100:.1f}%: {new_threshold:.6f}"
    )
    logger.info(f"Rescored range: [{rescored.min():.4f}, {rescored.max():.4f}]")

    # Save
    new_td = dict(td)
    new_td["threshold"] = new_threshold
    new_td["target_fpr"] = args.target_fpr
    new_td["calibrated_on"] = "live_benign_traffic"

    with open(args.models_dir / "threshold.json", "w") as f:
        json.dump(new_td, f, indent=2)

    with open(args.models_dir / "norm_stats.json", "w") as f:
        json.dump(new_norm_stats, f, indent=2)

    logger.info(f"\nSaved threshold.json and norm_stats.json to {args.models_dir}")
    logger.info("Restart the daemon to apply.")


if __name__ == "__main__":
    main()
