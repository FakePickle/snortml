"""
raw_capture_check.py
======================
Bypasses everything in snortml_inference — directly prints every flow
nfstream sees on the given interface as it happens, with no filtering,
no min_packets threshold, nothing. Used to verify capture is actually
working at all before debugging anything downstream.

Usage:
    python raw_capture_check.py "<interface>"
"""

import sys


def main():
    from nfstream import NFStreamer

    if len(sys.argv) < 2:
        print("Usage: python raw_capture_check.py <interface>")
        sys.exit(1)

    interface = sys.argv[1]
    print(f"Capturing on: {interface}")
    print("Browse something / ping something now. Printing every flow as it's seen...")
    print()

    streamer = NFStreamer(
        source=interface,
        idle_timeout=5,
        active_timeout=10,
        statistical_analysis=False,  # minimal overhead, just confirm capture works
    )

    count = 0
    for flow in streamer:
        count += 1
        print(
            f"[{count}] {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} "
            f"proto={flow.protocol} pkts={flow.bidirectional_packets} "
            f"bytes={flow.bidirectional_bytes}"
        )
        if count >= 30:
            print("\n30 flows captured — capture is working. Stopping.")
            break


if __name__ == "__main__":
    main()
