"""
Isolates whether statistical_analysis=True is what's breaking capture.
Run this exactly like raw_capture_check.py but with stat analysis ON,
matching stream_flows()'s actual config.
"""

import sys


def main():
    from nfstream import NFStreamer

    interface = sys.argv[1] if len(sys.argv) > 1 else None
    if not interface:
        print("Usage: python stat_analysis_check.py <interface>")
        sys.exit(1)

    print(f"Capturing on: {interface}  (statistical_analysis=True, splt_analysis=0)")
    print("Browse something now...")
    print()

    streamer = NFStreamer(
        source=interface,
        idle_timeout=5,
        active_timeout=10,
        statistical_analysis=True,
        splt_analysis=0,
    )

    count = 0
    for flow in streamer:
        count += 1
        print(
            f"[{count}] {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} "
            f"proto={flow.protocol} pkts={flow.bidirectional_packets}"
        )
        if count >= 15:
            print("\nWorking — statistical_analysis=True is NOT the problem.")
            break


if __name__ == "__main__":
    main()
