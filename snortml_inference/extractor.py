"""
snortml_inference/extractor.py
================================
Bridges nfstream's NFlow objects to the feature dict format
expected by the SnortML+ preprocessor.

nfstream extracts CICIDS-compatible flow features automatically.
We map nfstream field names → CICIoT2023 column names.

Cross-platform: works on Windows (Npcap), Linux, and RPi.
"""

import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# nfstream → CICIoT2023 feature name mapping
# nfstream uses slightly different names for the same concepts
# ---------------------------------------------------------------------------
NFSTREAM_TO_CICIOT = {
    # Timing
    "bidirectional_duration_ms": "flow_duration",
    # Packet counts / sizes
    "bidirectional_packets": "Number",
    "bidirectional_bytes": "Tot sum",
    "src2dst_bytes": "Tot size",
    # Header
    "ip_version": "IPv",
    "protocol": "Protocol Type",
    # TCP flags — bidirectional
    "bidirectional_syn_packets": "syn_count",
    "bidirectional_ack_packets": "ack_count",
    "bidirectional_fin_packets": "fin_count",
    "bidirectional_rst_packets": "rst_count",
    "bidirectional_psh_packets": "psh_flag_number",
    "bidirectional_urg_packets": "urg_count",
    # Stats
    "bidirectional_mean_ps": "AVG",
    "bidirectional_stddev_ps": "Std",
    "bidirectional_min_ps": "Min",
    "bidirectional_max_ps": "Max",
    "bidirectional_mean_piat_ms": "IAT",
    # Rates
    "src2dst_bytes": "Srate",
    "dst2src_bytes": "Drate",
}

# Protocol number → CICIoT2023 protocol type string (kept for logging/PROTO_NAMES use)
PROTO_MAP = {
    6: "TCP",
    17: "UDP",
    1: "ICMP",
    2: "IGMP",
    58: "ICMPv6",
}

# Protocol categories the model's OHE was fit on (from preprocessor_meta.json
# "ohe_categories"). Any protocol number not in this set produces an
# all-zero one-hot vector — matching how the trained Preprocessor handles
# unseen categories at inference time.
PROTOCOL_OHE_CATEGORIES = [0, 1, 2, 6, 17, 58, 83]

# Application layer flags (set 1 if destination port matches)
PORT_FLAGS = {
    "HTTP": {80, 8080, 8000},
    "HTTPS": {443, 8443},
    "DNS": {53},
    "SSH": {22},
    "SMTP": {25, 465, 587},
    "Telnet": {23},
    "IRC": {6667, 6668, 6669},
    "DHCP": {67, 68},
}


def flow_to_features(flow) -> Optional[dict]:
    """
    Convert an nfstream NFlow object to a CICIoT2023 feature dict.

    Returns None if the flow should be skipped (too few packets etc).
    """
    try:
        features = {}

        # Basic stats
        features["flow_duration"] = float(flow.bidirectional_duration_ms or 0)
        features["Number"] = float(flow.bidirectional_packets or 0)
        features["Tot sum"] = float(flow.bidirectional_bytes or 0)
        features["Tot size"] = float(flow.src2dst_bytes or 0)

        # Rates (bytes per ms → scale to reasonable range)
        dur = max(flow.bidirectional_duration_ms or 1, 1)
        features["Rate"] = float(flow.bidirectional_bytes or 0) / dur * 1000
        features["Srate"] = float(flow.src2dst_bytes or 0) / dur * 1000
        features["Drate"] = float(flow.dst2src_bytes or 0) / dur * 1000

        # Packet size stats
        features["Min"] = float(flow.bidirectional_min_ps or 0)
        features["Max"] = float(flow.bidirectional_max_ps or 0)
        features["AVG"] = float(flow.bidirectional_mean_ps or 0)
        features["Std"] = float(flow.bidirectional_stddev_ps or 0)

        # Inter-arrival time
        features["IAT"] = float(flow.bidirectional_mean_piat_ms or 0)

        # Header length & TTL — CONFIRMED via direct inspection (list_flow_attrs.py)
        # that this nfstream build exposes NEITHER attribute at the flow level
        # (99 total attributes checked, none containing "ttl" or "header"/"ip_size").
        # Earlier getattr() fallbacks to 20.0/40.0/64.0 were fabricated constants,
        # not real measurements — they silently differed from the training median
        # on every single flow, contributing a small but universal reconstruction
        # error that pushed otherwise-normal traffic just over threshold.
        #
        # Using the TRAINING FILL VALUE here instead is more honest: it's the
        # single best constant guess given we have no real per-flow data, and
        # minimizes (rather than fabricates) the reconstruction penalty. This is
        # not a substitute for real signal — if these features matter for your
        # use case, the actual fix is extracting raw IP/TCP headers directly via
        # scapy or similar instead of relying on nfstream's flow-level summary.
        features["Header_Length"] = 27.2  # training fill value (median)
        features["Time_To_Live"] = 83.1  # training fill value (median)
        # IPv — CICIoT2023's "IPv" column is a BINARY flag (clip range [0,1],
        # fill value ~1.0), not the literal IP version number. Sending raw
        # 4.0/6.0 here was wildly out-of-distribution on every single flow
        # (the StandardScaler was never fit on values outside [0,1] for this
        # column), inflating reconstruction error universally regardless of
        # whether the underlying traffic was normal. This was the dominant
        # cause of near-100% alert rate observed in testing.
        features["IPv"] = 1.0

        # TCP flags
        features["syn_flag_number"] = float(min(flow.bidirectional_syn_packets or 0, 1))
        features["ack_flag_number"] = float(min(flow.bidirectional_ack_packets or 0, 1))
        features["fin_flag_number"] = float(min(flow.bidirectional_fin_packets or 0, 1))
        features["rst_flag_number"] = float(min(flow.bidirectional_rst_packets or 0, 1))
        features["psh_flag_number"] = float(min(flow.bidirectional_psh_packets or 0, 1))
        features["urg_count"] = float(flow.bidirectional_urg_packets or 0)
        features["cwr_flag_number"] = 0.0
        features["ece_flag_number"] = 0.0

        features["syn_count"] = float(flow.bidirectional_syn_packets or 0)
        features["ack_count"] = float(flow.bidirectional_ack_packets or 0)
        features["fin_count"] = float(flow.bidirectional_fin_packets or 0)
        features["rst_count"] = float(flow.bidirectional_rst_packets or 0)

        # Protocol type — must be ONE-HOT ENCODED to match training.
        # The Preprocessor fit OHE on the raw numeric protocol value
        # (see preprocessor_meta.json: ohe_categories["Protocol Type"]
        # = [0, 1, 2, 6, 17, 58, 83]), producing columns named
        # "Protocol Type_<num>". A single string field here would silently
        # never match any of those columns, zeroing out all protocol signal.
        proto_num = flow.protocol or 0
        for cat in PROTOCOL_OHE_CATEGORIES:
            features[f"Protocol Type_{cat}"] = 1.0 if proto_num == cat else 0.0

        # Application layer binary flags
        dst_port = flow.dst_port or 0
        for app, ports in PORT_FLAGS.items():
            features[app] = 1.0 if dst_port in ports else 0.0

        # TCP/UDP/etc binary flags
        features["TCP"] = 1.0 if proto_num == 6 else 0.0
        features["UDP"] = 1.0 if proto_num == 17 else 0.0
        features["ICMP"] = 1.0 if proto_num == 1 else 0.0
        features["IGMP"] = 1.0 if proto_num == 2 else 0.0
        features["ARP"] = 0.0
        features["LLC"] = 0.0

        # Duration (same as flow_duration in ms)
        features["Duration"] = features["flow_duration"]

        # Statistical features
        features["Magnitue"] = float(features["AVG"] * features["Number"]) ** 0.5
        features["Radius"] = float(features["Std"] * features["Number"]) ** 0.5
        features["Covariance"] = float(features["Std"] ** 2)
        features["Variance"] = float(features["Std"] ** 2)
        features["Weight"] = float(features["Number"] / max(dur, 1))

        return features

    except Exception as e:
        logger.debug(f"Feature extraction failed: {e}")
        return None


def stream_flows(
    interface: str,
    min_packets: int = 5,
    idle_timeout: int = 10,
    active_timeout: int = 60,
) -> Iterator[dict]:
    """
    Yield feature dicts from live traffic on the given interface.

    Args:
        interface:      Network interface name (e.g. "eth0", "Wi-Fi")
        min_packets:    Skip flows with fewer packets (reduces noise)
        idle_timeout:   Export flow after N seconds of inactivity
        active_timeout: Force export after N seconds regardless
    """
    try:
        from nfstream import NFStreamer
    except ImportError:
        raise ImportError(
            "nfstream not installed. Run: pip install nfstream\n"
            "Windows also requires Npcap: https://npcap.com/#download"
        )

    logger.info(f"Starting capture on interface: {interface}")
    logger.info(f"Min packets: {min_packets} | Idle timeout: {idle_timeout}s")

    streamer = NFStreamer(
        source=interface,
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
        statistical_analysis=True,
        splt_analysis=0,
    )

    for flow in streamer:
        if flow.bidirectional_packets < min_packets:
            continue
        features = flow_to_features(flow)
        if features is not None:
            # Attach metadata for alert context
            features["_meta"] = {
                "src_ip": flow.src_ip,
                "dst_ip": flow.dst_ip,
                "src_port": flow.src_port,
                "dst_port": flow.dst_port,
                "protocol": flow.protocol,
                "packets": flow.bidirectional_packets,
                "bytes": flow.bidirectional_bytes,
                "duration_ms": flow.bidirectional_duration_ms,
            }
            yield features


def list_interfaces() -> None:
    """Print available network interfaces. Run this to find your interface name."""
    try:
        import socket

        from nfstream import NFStreamer

        print("\nAvailable interfaces:")
        # On Linux
        import os

        if os.path.exists("/sys/class/net"):
            for iface in os.listdir("/sys/class/net"):
                print(f"  {iface}")
        else:
            # Windows — nfstream will list them
            print("  Run: python -m snortml_inference --list-interfaces")
    except ImportError:
        print("nfstream not installed.")
