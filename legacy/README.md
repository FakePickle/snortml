# Legacy

The first version of the daemon, kept for reference. Nothing here is used
by the current package and these scripts no longer run against it.

- `models/` is the autoencoder trained on CICIoT2023 window features
  (Keras, ONNX fp32 and int8), with its scaler, threshold and Mahalanobis
  statistics.
- `extractor.py` mapped nfstream flows onto the CICIoT2023 column names.
- `recaliberation.py`, `diagnose.py`, `analysis.py`, `raw_capture.py`,
  `daemon_check.py`, `testflow.py` and `main.py` were one-off scripts for
  that model.

That model scored live traffic with features that did not mean the same
thing as the features it was trained on, which is why ordinary traffic was
flagged. The current package learns its baseline from the same flow records
it scores.
