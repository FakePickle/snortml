"""
Run this on the RPi from inside the snortml_inference_final directory.
Captures a handful of live flows, prints the exact feature dict produced
by extractor.py, and compares each value against the training fill_value
and clip range from preprocessor_meta.json — so we can see exactly which
features are out of distribution and by how much.

Usage:
    sudo uv run python diagnose_features.py <interface>

On Windows, nfstream spawns worker processes internally (via Python's
multiprocessing), which requires the entry point to be guarded by
`if __name__ == "__main__":` — otherwise Windows re-imports this module
in the child process and crashes before the real work starts.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    from nfstream import NFStreamer

    from snortml_inference.extractor import flow_to_features

    interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"

    with open("models/preprocessor_meta.json") as f:
        meta = json.load(f)
    fill_values = meta["fill_values"]
    clip_values = meta["clip_values"]

    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    streamer = NFStreamer(
        source=interface,
        statistical_analysis=True,
        idle_timeout=5,
        active_timeout=15,
    )

    count = 0
    for flow in streamer:
        if flow.bidirectional_packets < 3:
            continue
        feats = flow_to_features(flow)
        if feats is None:
            continue
        meta_info = {
            "src": f"{flow.src_ip}:{flow.src_port}",
            "dst": f"{flow.dst_ip}:{flow.dst_port}",
            "proto": flow.protocol,
            "pkts": flow.bidirectional_packets,
        }
        print(f"=== flow {count}  {meta_info} ===")
        for col in feature_cols:
            val = feats.get(col, "MISSING")
            train_fill = fill_values.get(col, "?")
            train_clip = clip_values.get(col, "?")
            flag = ""
            if (
                isinstance(val, (int, float))
                and isinstance(train_clip, list)
                and len(train_clip) == 2
            ):
                lo, hi = train_clip
                if val < lo or val > hi:
                    flag = "  <-- OUT OF TRAINING RANGE"
            print(
                f"  {col:20s} live={val!s:>14}  train_fill={train_fill!s:>10}  train_clip={train_clip}{flag}"
            )
        print()
        count += 1
        if count >= 5:
            break


if __name__ == "__main__":
    main()
