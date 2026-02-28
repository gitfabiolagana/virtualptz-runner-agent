#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _runner_post(base_url: str, api_key: str, path: str, payload: dict) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=8) as response:
        body = response.read().decode("utf-8", errors="ignore")
        return json.loads(body) if body else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Start ingest and launch tracking worker bound to returned job_id.")
    parser.add_argument("--runner-base-url", required=True, help="Runner URL, e.g. http://127.0.0.1:9001")
    parser.add_argument("--api-key", required=True, help="RUNNER_API_KEY")
    parser.add_argument("--session-id", required=True, help="Session id used for ingest/start")
    parser.add_argument("--source-uri-ref", required=True, help="Input source URI for ingest/start")
    parser.add_argument("--profile", choices=["quality", "balanced", "fluid"], default="balanced")
    parser.add_argument("--replica-index", type=int, default=0)
    parser.add_argument("--tracking-mode", choices=["sim", "file"], default="sim")
    parser.add_argument("--tracking-input-file", help="JSON input file used when --tracking-mode=file")
    parser.add_argument("--tracking-interval", type=float, default=1.0)
    parser.add_argument("--tracking-source", default="detector-worker")
    parser.add_argument("--worker-script", default="scripts/tracking_push_worker.py")
    parser.add_argument("--no-stop-on-exit", action="store_true", help="Do not stop ingest when launcher exits")
    args = parser.parse_args()

    if args.tracking_mode == "file" and not args.tracking_input_file:
        print("--tracking-input-file is required with --tracking-mode=file", file=sys.stderr)
        return 2

    try:
        start_data = _runner_post(
            args.runner_base_url,
            args.api_key,
            "/runner/ingest/start",
            {
                "session_id": args.session_id,
                "source_uri_ref": args.source_uri_ref,
                "profile": args.profile,
                "replica_index": args.replica_index,
            },
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"ingest start failed: {exc}", file=sys.stderr)
        return 1

    job_id = str(start_data.get("job_id", "")).strip()
    if not job_id:
        print(f"ingest start returned invalid payload: {start_data}", file=sys.stderr)
        return 1

    worker_cmd = [
        sys.executable,
        str(Path(args.worker_script).expanduser()),
        "--runner-base-url",
        args.runner_base_url,
        "--job-id",
        job_id,
        "--api-key",
        args.api_key,
        "--mode",
        args.tracking_mode,
        "--interval",
        str(args.tracking_interval),
        "--source",
        args.tracking_source,
    ]
    if args.tracking_mode == "file" and args.tracking_input_file:
        worker_cmd.extend(["--input-file", str(Path(args.tracking_input_file).expanduser())])

    print(f"ingest started: job_id={job_id}", flush=True)
    print(f"worker command: {' '.join(worker_cmd)}", flush=True)

    worker_proc = subprocess.Popen(worker_cmd)
    stop_requested = False

    def _signal_handler(signum, frame):  # type: ignore[no-untyped-def]
        nonlocal stop_requested
        stop_requested = True
        if worker_proc.poll() is None:
            worker_proc.terminate()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        return worker_proc.wait()
    finally:
        if worker_proc.poll() is None:
            worker_proc.terminate()
            worker_proc.wait(timeout=2)

        if not args.no_stop_on_exit:
            try:
                _runner_post(
                    args.runner_base_url,
                    args.api_key,
                    "/runner/ingest/stop",
                    {"job_id": job_id},
                )
                if stop_requested:
                    print(f"ingest stopped: job_id={job_id}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: failed to stop ingest {job_id}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
