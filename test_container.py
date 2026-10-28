"""
test_container.py
Send test_payload.txt to a running container instance and print/save the response.

Usage:
    python test_container.py
    python test_container.py --endpoint raw_inference
    python test_container.py --url http://localhost:3465 --conf 0.9 --output-map
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:3465"
DEFAULT_PAYLOAD = "test_payload.txt"
DEFAULT_OUTPUT = "test_output/response.json"

# Keys holding large base64 blobs — printed as a length instead of full text.
BLOB_KEYS = {"output_map", "logits", "bboxes"}


def parse_args():
    parser = argparse.ArgumentParser(description="Send test_payload.txt to a running PGCView container.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the running container.")
    parser.add_argument("--endpoint", default="full_pipeline", choices=["full_pipeline", "raw_inference"])
    parser.add_argument("--payload", default=DEFAULT_PAYLOAD, help="Path to a base64-encoded image payload file.")
    parser.add_argument("--conf", type=float, default=0.85, help="Detection confidence threshold.")
    parser.add_argument("--marker-type", default="all", choices=["all", "marker", "quadrat"],
                         help="/full_pipeline only.")
    parser.add_argument("--logits", action="store_true", help="/full_pipeline only: include raw logits in debug data.")
    parser.add_argument("--bboxes", action="store_true", help="/full_pipeline only: include raw bboxes in debug data.")
    parser.add_argument("--output-map", action="store_true", help="/full_pipeline only: include the segmentation output map.")
    parser.add_argument("--api-key", default=None, help="Optional bearer token, e.g. when pointed at a remote endpoint.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Where to save the full JSON response.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--wait-timeout", type=float, default=120.0,
                         help="Seconds to poll /ping for a healthy response before giving up.")
    parser.add_argument("--wait-interval", type=float, default=2.0,
                         help="Seconds between /ping attempts while waiting.")
    return parser.parse_args()


def wait_until_healthy(url, timeout, interval, headers=None):
    """
    Poll {url}/ping until it returns 200, 
    or raise TimeoutError after `timeout` seconds.
    """
    deadline = time.monotonic() + timeout
    last_error = None
    attempt = 1
    while time.monotonic() < deadline:
        try:
            print(f"Attempt {attempt}: GET {url}/ping")
            attempt += 1
            ping = requests.get(f"{url}/ping", headers=headers, timeout=interval)

            print(ping.text)
            if ping.status_code == 200:
                print(ping.status_code)
                return
            last_error = f"HTTP {ping.status_code}"
            print(last_error)
        except requests.RequestException as e:
            last_error = str(e)
        time.sleep(interval)
    raise TimeoutError(f"{url}/ping did not return 200 within {timeout}s (last: {last_error})")


def summarize(obj):
    """
    Recursively replace known base64 blob fields with their length for console printing.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in BLOB_KEYS and isinstance(v, str):
                out[k] = f"<base64 string, {len(v)} chars>"
            else:
                out[k] = summarize(v)
        return out
    if isinstance(obj, list):
        return [summarize(v) for v in obj]
    return obj


def main():
    args = parse_args()

    payload_path = Path(args.payload)
    if not payload_path.exists():
        print(f"Payload file not found: {payload_path}", file=sys.stderr)
        sys.exit(1)
    b64_str = payload_path.read_text().strip()

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    # Poll /ping until the container is healthy (models loaded), or bail after --wait-timeout.
    print(f"Waiting for {args.url}/ping to become healthy (timeout={args.wait_timeout}s)...")
    try:
        wait_until_healthy(args.url, args.wait_timeout, args.wait_interval, headers=headers)
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print("Container is healthy.")

    if args.endpoint == "raw_inference":
        params = {"conf": args.conf}
    else:
        params = {
            "conf": args.conf,
            "marker_type": args.marker_type,
            "logits": args.logits,
            "bboxes": args.bboxes,
            "output_map": args.output_map,
        }

    url = f"{args.url}/{args.endpoint}"
    print(f"POST {url} params={params}")

    try:
        response = requests.post(url, headers=headers, params=params, json={"b64_str": b64_str}, timeout=args.timeout)
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Status: {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        print("Response body was not valid JSON:")
        print(response.text)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Full response saved to {out_path}")

    print(json.dumps(summarize(data), indent=2))

    if not response.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
