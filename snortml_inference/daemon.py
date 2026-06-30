"""
snortml_inference/daemon.py
============================
Main inference loop. Captures flows, scores them, emits alerts.

Architecture note: capture and scoring run on SEPARATE THREADS, connected
by a thread-safe queue. This is required on Windows — calling onnxruntime's
InferenceSession.run() directly inside nfstream's generator loop (when
statistical_analysis=True spins up nfstream's own internal worker process)
silently hangs with no exception raised. Moving scoring to the main thread,
fed by a queue from a dedicated capture thread, avoids whatever execution
context conflict causes that hang.

Run as:
    python -m snortml_inference
    python -m snortml_inference --interface eth0
    python -m snortml_inference --list-interfaces
"""

import logging
import queue
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_running = True

# Sentinel placed on the queue to signal the capture thread has stopped
_SENTINEL = object()


def _handle_sigint(sig, frame):
    global _running
    print("\n[SnortML+] Shutting down...")
    _running = False


def run(config_path: str = "config.yaml") -> None:
    """Main daemon loop."""
    config = _load_config(config_path)
    _setup_logging(config)

    logger.info("=" * 60)
    logger.info("SnortML+ Inference Daemon")
    logger.info("=" * 60)

    # Load scorer
    from snortml_inference.alerter import Alerter
    from snortml_inference.scorer import AnomalyScorer

    models_dir = Path(config_path).parent / config["models_dir"]
    scorer = AnomalyScorer(str(models_dir), n_threads=config["performance"]["threads"])
    alerter = Alerter(config)

    # score_override lets the operator run with a different cutoff than the
    # model's calibrated threshold without retraining/recalibrating — useful
    # as a stopgap when the calibrated threshold (tuned on training data)
    # doesn't match observed live-traffic FPR. null = use scorer.threshold.
    score_override = config["alerts"].get("score_override")
    if score_override is not None:
        logger.info(
            f"score_override active: using {score_override:.4f} instead of "
            f"calibrated threshold {scorer.threshold:.4f}"
        )
        effective_threshold = float(score_override)
    else:
        effective_threshold = scorer.threshold

    interface = config["interface"]
    batch_size = config["capture"]["batch_size"]
    min_packets = config["capture"]["min_packets"]
    idle_timeout = config["capture"]["idle_timeout"]
    active_timeout = config["capture"]["active_timeout"]

    logger.info(f"Interface : {interface}")
    logger.info(
        f"Threshold : {effective_threshold:.4f}"
        + (" (override)" if score_override is not None else " (calibrated)")
    )
    logger.info(f"Batch size: {batch_size}")
    logger.info("Listening for flows... (Ctrl+C to stop)")
    logger.info("=" * 60)

    signal.signal(signal.SIGINT, _handle_sigint)

    # Flows cross from the capture thread to the main (scoring) thread
    # through this queue. maxsize provides backpressure so capture doesn't
    # run unboundedly ahead of scoring if scoring is briefly slow.
    flow_queue: "queue.Queue" = queue.Queue(maxsize=batch_size * 10)

    capture_thread = threading.Thread(
        target=_capture_worker,
        args=(interface, min_packets, idle_timeout, active_timeout, flow_queue),
        daemon=True,
        name="capture-worker",
    )
    capture_thread.start()

    buffer = []
    buffer_meta = []
    total_flows = 0
    total_alerts = 0
    t_start = time.time()
    last_flush = time.time()
    max_batch_wait_sec = config["capture"].get("max_batch_wait_sec", 5)

    try:
        while _running:
            try:
                item = flow_queue.get(timeout=1.0)
            except queue.Empty:
                # No flow arrived in the last second. Still check whether
                # it's time to flush a partial batch on the time-based
                # trigger below, then loop and re-check _running.
                item = None

            if item is _SENTINEL:
                # Capture thread ended (error or interface closed)
                break

            if item is not None:
                features = item
                meta = features.pop("_meta", {})
                buffer.append(features)
                buffer_meta.append(meta)

            time_to_flush = (time.time() - last_flush) >= max_batch_wait_sec
            count_to_flush = len(buffer) >= batch_size

            if buffer and (count_to_flush or time_to_flush):
                _, total_flows, total_alerts = _score_and_alert(
                    buffer,
                    buffer_meta,
                    scorer,
                    alerter,
                    total_flows,
                    total_alerts,
                    effective_threshold,
                )
                buffer = []
                buffer_meta = []
                last_flush = time.time()

        # Score any remaining flows
        if buffer:
            _, total_flows, total_alerts = _score_and_alert(
                buffer,
                buffer_meta,
                scorer,
                alerter,
                total_flows,
                total_alerts,
                effective_threshold,
            )

    except Exception as e:
        print(f"[DAEMON ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        logger.error(f"Daemon error: {e}")
    finally:
        elapsed = time.time() - t_start
        logger.info(f"\nSession summary:")
        logger.info(f"  Runtime    : {elapsed:.1f}s")
        logger.info(f"  Flows seen : {total_flows:,}")
        logger.info(f"  Alerts     : {total_alerts:,}")
        if total_flows > 0:
            logger.info(f"  Alert rate : {total_alerts / total_flows * 100:.2f}%")
        alerter.summary()


def _capture_worker(
    interface: str,
    min_packets: int,
    idle_timeout: int,
    active_timeout: int,
    flow_queue: "queue.Queue",
) -> None:
    """
    Runs on a dedicated thread. Pulls flows from nfstream and pushes them
    onto the queue for the main thread to score. Keeps nfstream's execution
    context fully isolated from the ONNX scoring calls.
    """
    from snortml_inference.extractor import stream_flows

    try:
        for features in stream_flows(
            interface,
            min_packets=min_packets,
            idle_timeout=idle_timeout,
            active_timeout=active_timeout,
        ):
            if not _running:
                break
            flow_queue.put(features)
    except Exception as e:
        print(f"[CAPTURE ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        logger.error(f"Capture thread error: {e}")
    finally:
        flow_queue.put(_SENTINEL)


def _score_and_alert(
    buffer, buffer_meta, scorer, alerter, total_flows, total_alerts, effective_threshold
):
    """
    Score a batch and emit alerts. Always runs on the main thread.

    is_anomaly is RE-DERIVED from effective_threshold rather than trusting
    scorer.score_batch()'s own is_anomaly array — score_batch() always
    compares against scorer.threshold (the calibrated value baked into the
    model artifact), which doesn't know about a config-level score_override.
    """
    try:
        X = scorer.preprocess_batch(buffer)
        _, scores = scorer.score_batch(X)
        is_anomaly = scores > effective_threshold

        for i, (anomaly, score) in enumerate(zip(is_anomaly, scores)):
            total_flows += 1
            if anomaly:
                total_alerts += 1
                # Re-attach metadata for alert context
                buffer[i]["_meta"] = buffer_meta[i]
                alerter.alert(buffer[i], float(score), effective_threshold)

    except Exception as e:
        print(f"[SCORING ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        logger.error(f"Scoring error: {e}")

    return [], total_flows, total_alerts


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper())
    log_file = log_cfg.get("file")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
