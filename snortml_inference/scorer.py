"""
snortml_inference/scorer.py
============================
Loads all model artifacts and scores flow feature vectors.
No TensorFlow dependency — pure onnxruntime + numpy.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import onnxruntime as rt

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """
    Loads the trained SnortML+ artifacts and scores flow vectors.

    Required files in models_dir:
        snortml_autoencoder_int8.onnx (or _fp32.onnx as fallback)
        scaler.joblib
        feature_cols.json
        preprocessor_meta.json
        threshold.json   (must include mse_norm_lo/hi, mahal_norm_lo/hi)
        mahal_mean.npy
        mahal_cov_inv.npy
    """

    def __init__(self, models_dir: str, n_threads: int = 4):
        self.models_dir = Path(models_dir)
        self._load_artifacts(n_threads)
        logger.info(
            f"AnomalyScorer ready | "
            f"features={len(self.feature_cols)} | "
            f"threshold={self.threshold:.4f}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score a batch of preprocessed flow vectors.

        Args:
            X: float32 array of shape (N, n_features) — already scaled

        Returns:
            (is_anomaly, scores) — bool array + float array
        """
        X = X.astype(np.float32)
        X_recon = self._session.run(None, {self._input_name: X})[0]
        err = X - X_recon

        # MSE per sample
        mse = np.mean(err**2, axis=1)

        # Mahalanobis distance
        diff = err - self.mahal_mean
        mahal = np.einsum("ij,jk,ik->i", diff, self.mahal_cov_inv, diff)
        mahal = np.sqrt(np.maximum(mahal, 0))

        # Normalise using fixed anchors from the benign calibration set.
        # clip(0, None) floors at 0 but allows scores above 1.0 — anomalies
        # SHOULD exceed the benign range; that excess is the signal the
        # threshold acts on. Must match model.py's _compute_scores exactly,
        # or inference will diverge from what the threshold was calibrated for.
        eps = 1e-8
        mse_norm = np.clip(
            (mse - self.norm_stats["mse_lo"])
            / (self.norm_stats["mse_hi"] - self.norm_stats["mse_lo"] + eps),
            0,
            None,
        )
        mahal_norm = np.clip(
            (mahal - self.norm_stats["mahal_lo"])
            / (self.norm_stats["mahal_hi"] - self.norm_stats["mahal_lo"] + eps),
            0,
            None,
        )

        scores = (
            self.score_weights["mse"] * mse_norm
            + self.score_weights["mahal"] * mahal_norm
        ).astype(np.float32)

        return scores > self.threshold, scores

    def preprocess(self, raw_features: dict) -> np.ndarray:
        """
        Convert a raw flow feature dict → scaled numpy array.

        Args:
            raw_features: dict mapping feature name → value
                          (as produced by extractor.py)

        Returns:
            float32 array of shape (1, n_features)
        """
        row = np.zeros(len(self.feature_cols), dtype=np.float32)

        for i, col in enumerate(self.feature_cols):
            val = raw_features.get(col, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = self.fill_values.get(col, 0.0)
            if np.isinf(val):
                val = self.fill_values.get(col, 0.0)
            # Clip to training range
            if col in self.clip_values:
                lo, hi = self.clip_values[col]
                val = max(lo, min(hi, val))
            row[i] = val

        X = row.reshape(1, -1)
        return self.scaler.transform(X).astype(np.float32)

    def preprocess_batch(self, flow_list: list) -> np.ndarray:
        """Preprocess a list of flow dicts → (N, n_features) array."""
        rows = np.vstack([self.preprocess(f) for f in flow_list])
        return rows

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_artifacts(self, n_threads: int) -> None:
        m = self.models_dir

        # ONNX session
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_path = m / "snortml_autoencoder_int8.onnx"
        if not onnx_path.exists():
            # Fallback to FP32
            onnx_path = m / "snortml_autoencoder_fp32.onnx"
            logger.warning("INT8 model not found, using FP32")

        self._session = rt.InferenceSession(str(onnx_path), opts)
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"ONNX model loaded: {onnx_path.name}")

        # Scaler
        self.scaler = joblib.load(m / "scaler.joblib")

        # Feature list
        with open(m / "feature_cols.json") as f:
            self.feature_cols = json.load(f)

        # Preprocessor metadata (fill values, clip values, OHE categories)
        with open(m / "preprocessor_meta.json") as f:
            meta = json.load(f)
        self.fill_values = meta.get("fill_values", {})
        raw_clip = meta.get("clip_values", {})
        self.clip_values = {k: tuple(v) for k, v in raw_clip.items()}
        self.ohe_categories = meta.get("ohe_categories", {})

        # Threshold + score weights + normalisation anchors
        # (all stored together in threshold.json — see model.py save())
        with open(m / "threshold.json") as f:
            td = json.load(f)
        self.threshold = td["threshold"]
        self.score_weights = {
            "mse": td.get("mse_weight", 0.6),
            "mahal": td.get("mahal_weight", 0.4),
        }
        self.norm_stats = {
            "mse_lo": td.get("mse_norm_lo"),
            "mse_hi": td.get("mse_norm_hi"),
            "mahal_lo": td.get("mahal_norm_lo"),
            "mahal_hi": td.get("mahal_norm_hi"),
        }
        if any(v is None for v in self.norm_stats.values()):
            raise RuntimeError(
                "threshold.json is missing normalisation anchors "
                "(mse_norm_lo/hi, mahal_norm_lo/hi). This model artifact "
                "predates the fixed-range normalisation fix — retrain or "
                "re-run calibrate_threshold() before deploying."
            )

        # Mahalanobis params
        self.mahal_mean = np.load(str(m / "mahal_mean.npy"))
        self.mahal_cov_inv = np.load(str(m / "mahal_cov_inv.npy"))
