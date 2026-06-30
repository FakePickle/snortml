# SnortML+ Inference Daemon

Lightweight anomaly detection daemon. No TensorFlow — just ONNX + nfstream.
Works on Windows, Linux, and Raspberry Pi 4.

---

## Setup

### Windows
```powershell
# 1. Install Npcap (required for packet capture)
#    https://npcap.com/#download — install with "WinPcap API compatibility" checked

# 2. Install Python dependencies
pip install nfstream onnxruntime scikit-learn joblib pyyaml

# 3. Find your interface name
python -m snortml_inference --list-interfaces

# 4. Update config.yaml → set interface: "Wi-Fi" or "Ethernet"
```

### Linux / Arch
```bash
sudo pacman -S libpcap  # or: sudo apt install libpcap-dev
pip install nfstream onnxruntime scikit-learn joblib pyyaml
```

### Raspberry Pi 4
```bash
sudo apt install libpcap-dev
pip install nfstream onnxruntime scikit-learn joblib pyyaml
```

---

## Directory structure
```
snortml_inference/
├── config.yaml              ← edit this first
├── requirements.txt
├── models/                  ← copy from training server
│   ├── snortml_autoencoder_int8.onnx
│   ├── scaler.joblib
│   ├── threshold.json
│   ├── feature_cols.json
│   ├── preprocessor_meta.json
│   ├── mahal_mean.npy
│   ├── mahal_cov_inv.npy
│   └── norm_stats.json
├── alerts/                  ← created automatically
│   └── snortml_alerts.log
└── snortml_inference/
    ├── __init__.py
    ├── __main__.py
    ├── daemon.py
    ├── extractor.py
    ├── scorer.py
    └── alerter.py
```

---

## Running

```bash
# List interfaces first
python -m snortml_inference --list-interfaces

# Run with config.yaml
python -m snortml_inference

# Override interface from command line
python -m snortml_inference --interface eth0
python -m snortml_inference --interface "Wi-Fi"

# Windows needs admin/sudo for raw packet capture
# Linux/RPi:
sudo python -m snortml_inference --interface eth0
```

---

## Alert format
```
[2026-06-22 12:30:00] [HIGH] SNORTML+ ANOMALY DETECTED | score=1.2341 threshold=0.7531 | 192.168.1.5:54321 → 8.8.8.8:443 proto=TCP pkts=12 bytes=4821 dur=230ms
```

Severity levels:
- **CRITICAL** — score > 3× threshold
- **HIGH** — score > 2× threshold  
- **MEDIUM** — score > 1.5× threshold
- **LOW** — score just above threshold

---

## Running as a service (RPi / Linux)

```bash
sudo nano /etc/systemd/system/snortml.service
```

```ini
[Unit]
Description=SnortML+ Anomaly Detector
After=network.target

[Service]
ExecStart=/usr/bin/python3 -m snortml_inference --interface eth0
WorkingDirectory=/home/pi/snortml_inference
Restart=always
User=root

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable snortml
sudo systemctl start snortml
sudo journalctl -u snortml -f
```
