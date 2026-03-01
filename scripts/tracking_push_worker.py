#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _state_from_confidence(confidence: float) -> str:
    if confidence < 0.30:
        return "lost"
    if confidence < 0.60:
        return "warning"
    return "stable"


def _post_tracking_update(base_url: str, api_key: str, payload: dict) -> dict:
    url = f"{base_url.rstrip('/')}/runner/tracking/update"
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        body = response.read().decode("utf-8", errors="ignore")
        return json.loads(body) if body else {}


def _simulated_payload(job_id: str, tick: int, source: str) -> dict:
    wave = math.sin((tick % 120) / 120.0 * math.tau)
    conf = _clamp(0.62 + (0.22 * wave) + random.uniform(-0.05, 0.05), 0.0, 1.0)
    occ = _clamp(0.45 - conf + random.uniform(-0.08, 0.08), 0.0, 1.0)
    scan_amp_x = 0.08 + (0.22 * (1.0 - conf)) + (0.18 * occ)
    scan_amp_y = 0.04 + (0.14 * (1.0 - conf)) + (0.12 * occ)
    target_x = _clamp(0.5 + (scan_amp_x * math.sin((tick % 240) / 240.0 * math.tau)), 0.0, 1.0)
    target_y = _clamp(0.5 + (scan_amp_y * math.sin(((tick + 47) % 300) / 300.0 * math.tau)), 0.0, 1.0)
    target_zoom = _clamp(1.05 + (1.25 * conf) - (0.4 * occ), 1.0, 4.0)
    targets: list[dict[str, object]] = []
    for idx in range(10):
        base_phase = ((tick + (idx * 13)) % 360) / 360.0 * math.tau
        px = _clamp(0.5 + (0.34 * math.sin(base_phase + (idx * 0.19))), 0.0, 1.0)
        py = _clamp(0.52 + (0.16 * math.sin((base_phase * 0.83) + (idx * 0.11))), 0.0, 1.0)
        pconf = _clamp(conf + random.uniform(-0.20, 0.15), 0.2, 0.99)
        targets.append(
            {
                "x": round(px, 4),
                "y": round(py, 4),
                "confidence": round(pconf, 4),
                "cls": "player",
            }
        )
    ball_x = _clamp(target_x + random.uniform(-0.07, 0.07), 0.0, 1.0)
    ball_y = _clamp(target_y + random.uniform(-0.05, 0.05), 0.0, 1.0)
    targets.append(
        {
            "x": round(ball_x, 4),
            "y": round(ball_y, 4),
            "confidence": round(_clamp(conf + 0.12, 0.35, 1.0), 4),
            "cls": "ball",
        }
    )
    return {
        "job_id": job_id,
        "tracking_confidence": round(conf, 4),
        "tracking_state": _state_from_confidence(conf),
        "fallback_active": conf < 0.42,
        "occlusion_level": round(occ, 4),
        "source": source,
        "target_x": round(target_x, 4),
        "target_y": round(target_y, 4),
        "target_zoom": round(target_zoom, 4),
        "targets": targets,
    }


def _file_payload(job_id: str, file_path: Path, source: str) -> dict:
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    conf = float(raw.get("tracking_confidence", 0.0))
    fallback = bool(raw.get("fallback_active", conf < 0.42))
    state = str(raw.get("tracking_state", _state_from_confidence(conf))).strip().lower()
    occ = float(raw.get("occlusion_level", 0.0))
    raw_targets = raw.get("targets")
    targets: list[dict[str, object]] = []
    if isinstance(raw_targets, list):
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            try:
                tx = _clamp(float(item.get("x")), 0.0, 1.0)
                ty = _clamp(float(item.get("y")), 0.0, 1.0)
            except (TypeError, ValueError):
                continue
            tconf = _clamp(float(item.get("confidence", conf)), 0.0, 1.0)
            targets.append(
                {
                    "x": round(tx, 4),
                    "y": round(ty, 4),
                    "confidence": round(tconf, 4),
                    "cls": str(item.get("cls", "player"))[:32],
                }
            )

    return {
        "job_id": job_id,
        "tracking_confidence": round(_clamp(conf, 0.0, 1.0), 4),
        "tracking_state": state,
        "fallback_active": fallback,
        "occlusion_level": round(_clamp(occ, 0.0, 1.0), 4),
        "source": str(raw.get("source", source)).strip().lower(),
        "target_x": (
            round(_clamp(float(raw["target_x"]), 0.0, 1.0), 4)
            if raw.get("target_x") is not None
            else None
        ),
        "target_y": (
            round(_clamp(float(raw["target_y"]), 0.0, 1.0), 4)
            if raw.get("target_y") is not None
            else None
        ),
        "target_zoom": (
            round(_clamp(float(raw["target_zoom"]), 1.0, 4.0), 4)
            if raw.get("target_zoom") is not None
            else None
        ),
        "tracking_mode_hint": raw.get("tracking_mode_hint"),
        "play_phase": raw.get("play_phase"),
        "ball_velocity_x": raw.get("ball_velocity_x"),
        "ball_velocity_y": raw.get("ball_velocity_y"),
        "possession_side": raw.get("possession_side"),
        "court_homography_ok": raw.get("court_homography_ok"),
        "targets": targets or None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Push tracking metrics to VirtualPTZ runner.")
    parser.add_argument("--runner-base-url", required=True, help="Runner base URL, e.g. http://127.0.0.1:9001")
    parser.add_argument("--job-id", required=True, help="Ingest job_id returned by /runner/ingest/start")
    parser.add_argument("--api-key", required=True, help="RUNNER_API_KEY")
    parser.add_argument("--mode", choices=["sim", "file"], default="sim", help="Data source mode")
    parser.add_argument("--input-file", help="JSON file path used when --mode=file")
    parser.add_argument("--interval", type=float, default=1.0, help="Push interval seconds")
    parser.add_argument("--source", default="detector-worker", help="Tracking source label")
    args = parser.parse_args()

    if args.mode == "file" and not args.input_file:
        print("--input-file is required in file mode", file=sys.stderr)
        return 2

    input_file = Path(args.input_file).expanduser() if args.input_file else None
    tick = 0
    while True:
        tick += 1
        try:
            if args.mode == "sim":
                payload = _simulated_payload(args.job_id, tick, args.source)
            else:
                if input_file is None or not input_file.exists():
                    raise FileNotFoundError(f"tracking input file not found: {input_file}")
                payload = _file_payload(args.job_id, input_file, args.source)
            result = _post_tracking_update(args.runner_base_url, args.api_key, payload)
            print(
                f"push ok conf={result.get('tracking_confidence')} state={result.get('tracking_state')} "
                f"fallback={result.get('fallback_active')}",
                flush=True,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"push failed: {exc}", file=sys.stderr, flush=True)
        except (json.JSONDecodeError, ValueError, FileNotFoundError) as exc:
            print(f"input error: {exc}", file=sys.stderr, flush=True)
        time.sleep(max(0.2, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
