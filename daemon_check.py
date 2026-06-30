"""
Replicates daemon.py's actual import order: AnomalyScorer (which imports
onnxruntime, joblib) gets loaded BEFORE extractor.stream_flows() is ever
called. Testing whether THAT ordering is what breaks nfstream's worker spawn.
"""

import sys


def main():
    interface = sys.argv[1] if len(sys.argv) > 1 else None
    if not interface:
        print("Usage: python import_order_check.py <interface>")
        sys.exit(1)

    print("Importing AnomalyScorer (onnxruntime, joblib) first...")
    sys.path.insert(0, ".")
    from snortml_inference.scorer import AnomalyScorer

    scorer = AnomalyScorer("models")
    print(f"Scorer loaded OK. threshold={scorer.threshold}")
    print()

    print("NOW importing nfstream and starting capture...")
    from nfstream import NFStreamer

    streamer = NFStreamer(
        source=interface,
        idle_timeout=5,
        active_timeout=10,
        statistical_analysis=True,
        splt_analysis=0,
    )

    print("Browse something now...")
    count = 0
    for flow in streamer:
        count += 1
        print(
            f"[{count}] {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} "
            f"pkts={flow.bidirectional_packets}"
        )
        if count >= 15:
            print("\nWorking even after scorer/onnxruntime import.")
            return


if __name__ == "__main__":
    main()
