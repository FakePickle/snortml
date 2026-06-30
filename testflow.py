"""
Prints every attribute available on a real NFlow object from your nfstream
version, so we can find the ACTUAL names for header size and TTL instead
of guessing. Run from snortml_inference_final directory.
"""

import sys


def main():
    from nfstream import NFStreamer

    interface = sys.argv[1] if len(sys.argv) > 1 else None
    if not interface:
        print("Usage: python list_flow_attrs.py <interface>")
        sys.exit(1)

    streamer = NFStreamer(
        source=interface,
        statistical_analysis=True,
        idle_timeout=5,
        active_timeout=10,
        splt_analysis=0,
    )

    for flow in streamer:
        attrs = sorted([a for a in dir(flow) if not a.startswith("_")])
        print(f"Total attributes: {len(attrs)}\n")
        for a in attrs:
            try:
                val = getattr(flow, a)
                if not callable(val):
                    print(f"  {a:40s} = {val}")
            except Exception:
                pass
        break  # just need one flow


if __name__ == "__main__":
    main()
