from __future__ import annotations

import os
import subprocess
import threading
import math
import random
import time
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


class ProbeRequest(BaseModel):
    source_uri_ref: str
    timeout_seconds: float = 6.0


class PreviewStartRequest(BaseModel):
    source_uri_ref: str
    profile: dict = Field(default_factory=dict)


class PreviewStopRequest(BaseModel):
    job_id: str


class IngestStartRequest(BaseModel):
    session_id: str
    source_uri_ref: str
    profile: str
    replica_index: int = 0


class IngestStopRequest(BaseModel):
    job_id: str


class JobStatusRequest(BaseModel):
    job_id: str


class TrackingStatusRequest(BaseModel):
    job_id: str


class TrackingUpdateRequest(BaseModel):
    job_id: str
    tracking_confidence: float = Field(ge=0.0, le=1.0)
    tracking_state: str = Field(default="warning", min_length=3, max_length=32)
    fallback_active: bool = False
    occlusion_level: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(default="external", min_length=2, max_length=32)


class PtzUpdateRequest(BaseModel):
    job_id: str
    center_x: float = Field(default=0.5, ge=0.0, le=1.0)
    center_y: float = Field(default=0.5, ge=0.0, le=1.0)
    zoom: float = Field(default=1.0, ge=1.0, le=4.0)


app = FastAPI(title="VirtualPTZ Runner Agent", version="0.2.0")
RUNNER_API_KEY = os.getenv("RUNNER_API_KEY", "").strip()
RUNNER_PUBLIC_BASE_URL = os.getenv("RUNNER_PUBLIC_BASE_URL", "http://127.0.0.1:9001").rstrip("/")
INGEST_OUTPUT_URI = os.getenv("RUNNER_INGEST_OUTPUT_URI", "").strip()
PREVIEW_DEFAULT_FPS = max(1, min(60, int(os.getenv("PREVIEW_MJPEG_FPS", "20"))))
PREVIEW_DEFAULT_SCALE = os.getenv("PREVIEW_MJPEG_SCALE", "640:-1").strip() or "640:-1"
PREVIEW_DEFAULT_QV = max(2, min(31, int(os.getenv("PREVIEW_MJPEG_Q", "8"))))
PREVIEW_LOW_LATENCY = os.getenv("PREVIEW_MJPEG_LOW_LATENCY", "1").strip() not in {"0", "false", "False"}
PTZ_MIN_APPLY_INTERVAL_SECONDS = max(0.1, float(os.getenv("RUNNER_PTZ_MIN_APPLY_INTERVAL_SECONDS", "0.35")))
PTZ_MIN_DELTA_XY = max(0.0, float(os.getenv("RUNNER_PTZ_MIN_DELTA_XY", "0.008")))
PTZ_MIN_DELTA_ZOOM = max(0.0, float(os.getenv("RUNNER_PTZ_MIN_DELTA_ZOOM", "0.015")))

preview_jobs: dict[str, dict] = {}
ingest_jobs: dict[str, dict] = {}
state_lock = threading.Lock()


def _mask_sensitive_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if not parsed.hostname:
        return uri
    host = parsed.hostname
    if parsed.port:
        host = f"{host}:{parsed.port}"
    if parsed.username:
        netloc = f"{parsed.username}:***@{host}"
    else:
        netloc = host
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _auth(header: str | None) -> None:
    if not RUNNER_API_KEY:
        return
    if not header or not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = header.removeprefix("Bearer ").strip()
    if token != RUNNER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid bearer token")


def _kill_process(proc: subprocess.Popen | None, timeout: float = 2.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1)


def _ffprobe_cmd(source_uri: str) -> list[str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
    ]
    if source_uri.startswith("rtsp://"):
        cmd.extend(["-rtsp_transport", "tcp"])
    cmd.append(source_uri)
    return cmd


def _is_file_like_source(source_uri: str) -> bool:
    uri = (source_uri or "").strip().lower()
    if uri.startswith("file://"):
        return True
    return "://" not in uri


def _ffmpeg_preview_cmd(
    source_uri: str,
    fps: int = PREVIEW_DEFAULT_FPS,
    scale: str = PREVIEW_DEFAULT_SCALE,
    q_v: int = PREVIEW_DEFAULT_QV,
    ptz: dict | None = None,
    start_seconds: float = 0.0,
) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if PREVIEW_LOW_LATENCY:
        cmd.extend(["-fflags", "nobuffer", "-flags", "low_delay", "-analyzeduration", "0", "-probesize", "32768"])
    if _is_file_like_source(source_uri):
        # Keep file previews at wall-clock speed (avoid fast-forward effect in MJPEG).
        cmd.extend(["-re"])
    if source_uri.startswith("rtsp://"):
        cmd.extend(["-rtsp_transport", "tcp"])
    elif source_uri.startswith("rtmp://"):
        cmd.extend(["-rw_timeout", "15000000"])
    vf = f"fps={fps},scale={scale}"
    if isinstance(ptz, dict):
        zoom = _clamp(float(ptz.get("zoom", 1.0)), 1.0, 4.0)
        cx = _clamp(float(ptz.get("center_x", 0.5)), 0.0, 1.0)
        cy = _clamp(float(ptz.get("center_y", 0.5)), 0.0, 1.0)
        if zoom > 1.01:
            vf = (
                f"crop=w=iw/{zoom:.4f}:h=ih/{zoom:.4f}:x=(iw-iw/{zoom:.4f})*{cx:.4f}:"
                f"y=(ih-ih/{zoom:.4f})*{cy:.4f},fps={fps},scale={scale}"
            )
    if _is_file_like_source(source_uri) and start_seconds > 0.001:
        cmd.extend(["-ss", f"{start_seconds:.3f}"])
    cmd.extend(["-i", source_uri, "-an", "-vf", vf, "-q:v", str(q_v), "-flush_packets", "1", "-f", "mjpeg", "pipe:1"])
    return cmd


def _ptz_filter(center_x: float, center_y: float, zoom: float) -> str:
    z = _clamp(zoom, 1.0, 4.0)
    cx = _clamp(center_x, 0.0, 1.0)
    cy = _clamp(center_y, 0.0, 1.0)
    if z <= 1.01:
        return "scale=1280:720"
    return f"crop=w=iw/{z:.4f}:h=ih/{z:.4f}:x=(iw-iw/{z:.4f})*{cx:.4f}:y=(ih-ih/{z:.4f})*{cy:.4f},scale=1280:720"


def _ffmpeg_ingest_cmd(source_uri: str, profile: str, ptz: dict[str, float] | None = None) -> list[str]:
    # Profile-specific transcode defaults; can be evolved per production policy.
    preset = {
        "quality": ["-preset", "slow", "-crf", "20"],
        "balanced": ["-preset", "veryfast", "-crf", "24"],
        "fluid": ["-preset", "ultrafast", "-crf", "28"],
    }.get(profile, ["-preset", "veryfast", "-crf", "24"])

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if source_uri.startswith("rtsp://"):
        cmd.extend(["-rtsp_transport", "tcp"])
    elif source_uri.startswith("rtmp://"):
        cmd.extend(["-rw_timeout", "15000000"])
    ptz_cfg = ptz or {}
    vf = _ptz_filter(
        float(ptz_cfg.get("center_x", 0.5)),
        float(ptz_cfg.get("center_y", 0.5)),
        float(ptz_cfg.get("zoom", 1.0)),
    )
    cmd.extend(["-i", source_uri, "-an", "-vf", vf, "-c:v", "libx264", *preset])

    if INGEST_OUTPUT_URI:
        cmd.extend(["-f", "flv", INGEST_OUTPUT_URI])
    else:
        cmd.extend(["-f", "null", "-"])
    return cmd


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _tracking_state(confidence: float, previous: str = "lost") -> str:
    if previous == "lost":
        if confidence >= 0.48:
            return "warning"
        return "lost"
    if previous == "stable":
        if confidence < 0.50:
            return "warning"
        return "stable"
    if confidence < 0.28:
        return "lost"
    if confidence >= 0.62:
        return "stable"
    return "warning"


def _fallback_target(confidence: float, current: bool) -> bool:
    if current:
        return confidence < 0.50
    return confidence < 0.42


def _init_tracking_state(profile: str) -> dict:
    base_conf = {"quality": 0.72, "balanced": 0.64, "fluid": 0.58}.get(profile, 0.62)
    return {
        "tracking_confidence": base_conf,
        "tracking_state": _tracking_state(base_conf, "warning"),
        "fallback_active": False,
        "occlusion_level": 0.0,
        "source": "runner-sim",
        "updated_at": datetime.utcnow().isoformat(),
    }


def _advance_tracking(job: dict) -> dict:
    tracking = dict(job.get("tracking", {}))
    confidence = float(tracking.get("tracking_confidence", 0.55))
    occlusion = float(tracking.get("occlusion_level", 0.0))
    prev_state = str(tracking.get("tracking_state", "warning"))
    prev_fallback = bool(tracking.get("fallback_active", False))
    profile = str(job.get("profile", "balanced"))
    profile_bias = {"quality": 0.06, "balanced": 0.0, "fluid": -0.04}.get(profile, 0.0)
    wave = math.sin((datetime.utcnow().timestamp() % 60) / 60.0 * math.tau)
    conf_delta = random.uniform(-0.045, 0.045) + (wave * 0.015) + profile_bias
    confidence = _clamp(confidence + conf_delta - (0.32 * occlusion), 0.0, 1.0)
    occlusion = _clamp(occlusion + random.uniform(-0.12, 0.2) - (confidence * 0.07), 0.0, 1.0)
    state = _tracking_state(confidence, previous=prev_state)
    fallback = _fallback_target(confidence, current=prev_fallback)
    tracking.update(
        {
            "tracking_confidence": round(confidence, 4),
            "tracking_state": state,
            "fallback_active": fallback,
            "occlusion_level": round(occlusion, 4),
            "source": tracking.get("source", "runner-sim"),
            "updated_at": datetime.utcnow().isoformat(),
        }
    )
    return tracking


def _start_ingest_watcher(job_id: str, proc: subprocess.Popen) -> None:
    def _watch() -> None:
        rc = proc.wait()
        with state_lock:
            job = ingest_jobs.get(job_id)
            if not job:
                return
            if job.get("proc") is not proc:
                # A newer process replaced this one (e.g. PTZ reconfigure).
                return
            if job["state"] == "stopped":
                return
            if rc == 0:
                job["state"] = "stopped"
            else:
                job["state"] = "failed"
                job["restart_count"] = int(job.get("restart_count", 0)) + 1
                job["last_error"] = f"ffmpeg exited with code {rc}"
            job["updated_at"] = datetime.utcnow().isoformat()

    t = threading.Thread(target=_watch, daemon=True)
    t.start()


@app.get("/health")
def health() -> dict:
    with state_lock:
        return {
            "status": "ok",
            "preview_jobs": len(preview_jobs),
            "ingest_jobs": len(ingest_jobs),
        }


@app.post("/runner/probe")
def probe(req: ProbeRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    cmd = _ffprobe_cmd(req.source_uri_ref)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1.0, req.timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="probe timeout") from None

    if proc.returncode != 0:
        raise HTTPException(status_code=502, detail="probe failed")

    details = [line for line in proc.stdout.splitlines() if line.strip()]
    return {"ok": True, "details": details}


@app.post("/runner/preview/start")
def preview_start(req: PreviewStartRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    job_id = str(uuid4())
    fps = int(req.profile.get("fps", PREVIEW_DEFAULT_FPS)) if isinstance(req.profile, dict) else PREVIEW_DEFAULT_FPS
    scale = str(req.profile.get("scale", PREVIEW_DEFAULT_SCALE)) if isinstance(req.profile, dict) else PREVIEW_DEFAULT_SCALE
    q_v = int(req.profile.get("q_v", req.profile.get("q", PREVIEW_DEFAULT_QV))) if isinstance(req.profile, dict) else PREVIEW_DEFAULT_QV
    fps = max(1, min(60, fps))
    q_v = max(2, min(31, q_v))
    scale = scale.strip() or PREVIEW_DEFAULT_SCALE
    ptz = req.profile.get("ptz") if isinstance(req.profile, dict) else None
    start_seconds = 0.0
    if isinstance(req.profile, dict):
        try:
            start_seconds = max(0.0, float(req.profile.get("start_seconds", 0.0)))
        except (TypeError, ValueError):
            start_seconds = 0.0
    cmd = _ffmpeg_preview_cmd(
        req.source_uri_ref,
        fps=fps,
        scale=scale,
        q_v=q_v,
        ptz=ptz,
        start_seconds=start_seconds,
    )
    with state_lock:
        preview_jobs[job_id] = {
            "job_id": job_id,
            "source_uri_ref": req.source_uri_ref,
            "command": cmd,
            "created_at": datetime.utcnow().isoformat(),
        }
    return {"job_id": job_id, "preview_url": f"{RUNNER_PUBLIC_BASE_URL}/runner/preview/stream/{job_id}"}


@app.get("/runner/preview/stream/{job_id}")
def preview_stream(job_id: str, authorization: str | None = Header(default=None)) -> StreamingResponse:
    _auth(authorization)
    with state_lock:
        job = preview_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="preview job not found")

    cmd = list(job["command"])

    def _iter_frames():
        proc: subprocess.Popen | None = None
        buffer = b""
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.stdout is None:
                raise HTTPException(status_code=500, detail="preview pipe unavailable")
            while True:
                chunk = proc.stdout.read(32768)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    start = buffer.find(JPEG_SOI)
                    if start < 0:
                        if len(buffer) > 2:
                            buffer = buffer[-2:]
                        break
                    end = buffer.find(JPEG_EOI, start + 2)
                    if end < 0:
                        if start > 0:
                            buffer = buffer[start:]
                        break
                    frame = buffer[start : end + 2]
                    buffer = buffer[end + 2 :]
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        + frame
                        + b"\r\n"
                    )
        finally:
            _kill_process(proc)
            with state_lock:
                preview_jobs.pop(job_id, None)

    return StreamingResponse(
        _iter_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/runner/preview/stop")
def preview_stop(req: PreviewStopRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    with state_lock:
        preview_jobs.pop(req.job_id, None)
    return {"stopped": True}


@app.post("/runner/ingest/start")
def ingest_start(req: IngestStartRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    job_id = str(uuid4())
    ptz_state = {"center_x": 0.5, "center_y": 0.5, "zoom": 1.0}
    cmd = _ffmpeg_ingest_cmd(req.source_uri_ref, req.profile, ptz=ptz_state)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError:
        raise HTTPException(status_code=500, detail="failed to start ffmpeg") from None

    with state_lock:
        ingest_jobs[job_id] = {
            "job_id": job_id,
            "session_id": req.session_id,
            "replica_index": req.replica_index,
            "profile": req.profile,
            "state": "running",
            "restart_count": 0,
            "last_error": None,
            "pid": proc.pid,
            "proc": proc,
            "masked_source": _mask_sensitive_uri(req.source_uri_ref),
            "source_uri_ref": req.source_uri_ref,
            "ptz": ptz_state,
            "last_ptz_apply_ts": time.monotonic(),
            "tracking": _init_tracking_state(req.profile),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
    _start_ingest_watcher(job_id, proc)
    return {"job_id": job_id, "state": "running"}


@app.post("/runner/ingest/stop")
def ingest_stop(req: IngestStopRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    with state_lock:
        job = ingest_jobs.get(req.job_id)
    if not job:
        return {"state": "stopped"}

    proc = job.get("proc")
    _kill_process(proc)
    with state_lock:
        job["state"] = "stopped"
        job["updated_at"] = datetime.utcnow().isoformat()
    return {"state": "stopped"}


@app.post("/runner/job/status")
def status(req: JobStatusRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    with state_lock:
        job = ingest_jobs.get(req.job_id)
        if not job:
            return {"job_id": req.job_id, "state": "unknown", "restart_count": 0, "last_error": None}
        proc = job.get("proc")
        if proc is not None and proc.poll() is not None and job["state"] == "running":
            rc = proc.returncode
            if rc == 0:
                job["state"] = "stopped"
            else:
                job["state"] = "failed"
                job["restart_count"] = int(job.get("restart_count", 0)) + 1
                job["last_error"] = f"ffmpeg exited with code {rc}"
            job["updated_at"] = datetime.utcnow().isoformat()

        return {
            "job_id": req.job_id,
            "state": job["state"],
            "restart_count": int(job.get("restart_count", 0)),
            "last_error": job.get("last_error"),
        }


@app.post("/runner/tracking/update")
def tracking_update(req: TrackingUpdateRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    with state_lock:
        job = ingest_jobs.get(req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="tracking job not found")
        tracking = dict(job.get("tracking", {}))
        tracking.update(
            {
                "tracking_confidence": round(req.tracking_confidence, 4),
                "tracking_state": req.tracking_state.strip().lower(),
                "fallback_active": req.fallback_active,
                "occlusion_level": round(req.occlusion_level, 4),
                "source": req.source.strip().lower(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
        job["tracking"] = tracking
        job["updated_at"] = datetime.utcnow().isoformat()
        return {"job_id": req.job_id, **tracking}


@app.post("/runner/tracking/status")
def tracking_status(req: TrackingStatusRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    with state_lock:
        job = ingest_jobs.get(req.job_id)
        if not job:
            return {
                "job_id": req.job_id,
                "tracking_confidence": 0.0,
                "tracking_state": "lost",
                "fallback_active": True,
                "occlusion_level": 1.0,
                "source": "runner",
                "available": False,
            }
        if job.get("state") == "running":
            if str(job.get("tracking", {}).get("source", "runner-sim")) == "runner-sim":
                job["tracking"] = _advance_tracking(job)
        tracking = dict(job.get("tracking", {}))
        return {
            "job_id": req.job_id,
            "tracking_confidence": float(tracking.get("tracking_confidence", 0.0)),
            "tracking_state": str(tracking.get("tracking_state", "lost")),
            "fallback_active": bool(tracking.get("fallback_active", True)),
            "occlusion_level": float(tracking.get("occlusion_level", 1.0)),
            "source": str(tracking.get("source", "runner")),
            "available": True,
            "updated_at": tracking.get("updated_at"),
        }


@app.post("/runner/ptz/update")
def ptz_update(req: PtzUpdateRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    now = time.monotonic()
    with state_lock:
        job = ingest_jobs.get(req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="ptz job not found")
        if job.get("state") != "running":
            raise HTTPException(status_code=409, detail="ptz update requires a running ingest job")

        old_proc = job.get("proc")
        source_uri_ref = str(job.get("source_uri_ref", "")).strip()
        profile = str(job.get("profile", "balanced"))
        if not source_uri_ref:
            raise HTTPException(status_code=500, detail="ingest job missing source_uri_ref")

        ptz_state = {
            "center_x": round(_clamp(req.center_x, 0.0, 1.0), 4),
            "center_y": round(_clamp(req.center_y, 0.0, 1.0), 4),
            "zoom": round(_clamp(req.zoom, 1.0, 4.0), 4),
        }
        prev_ptz = job.get("ptz") or {"center_x": 0.5, "center_y": 0.5, "zoom": 1.0}
        prev_x = float(prev_ptz.get("center_x", 0.5))
        prev_y = float(prev_ptz.get("center_y", 0.5))
        prev_zoom = float(prev_ptz.get("zoom", 1.0))
        delta_x = abs(ptz_state["center_x"] - prev_x)
        delta_y = abs(ptz_state["center_y"] - prev_y)
        delta_zoom = abs(ptz_state["zoom"] - prev_zoom)
        is_tiny_delta = delta_x < PTZ_MIN_DELTA_XY and delta_y < PTZ_MIN_DELTA_XY and delta_zoom < PTZ_MIN_DELTA_ZOOM
        last_applied_ts = float(job.get("last_ptz_apply_ts", 0.0))
        in_cooldown = (now - last_applied_ts) < PTZ_MIN_APPLY_INTERVAL_SECONDS
        if is_tiny_delta or in_cooldown:
            return {
                "job_id": req.job_id,
                "state": "running",
                "ptz": {
                    "center_x": round(prev_x, 4),
                    "center_y": round(prev_y, 4),
                    "zoom": round(prev_zoom, 4),
                },
                "applied": False,
                "reason": "tiny_delta" if is_tiny_delta else "cooldown",
            }
        cmd = _ffmpeg_ingest_cmd(source_uri_ref, profile, ptz=ptz_state)
        try:
            new_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except OSError:
            raise HTTPException(status_code=500, detail="failed to restart ffmpeg for ptz update") from None

        job["proc"] = new_proc
        job["pid"] = new_proc.pid
        job["ptz"] = ptz_state
        job["last_ptz_apply_ts"] = now
        job["updated_at"] = datetime.utcnow().isoformat()

    _kill_process(old_proc)
    _start_ingest_watcher(req.job_id, new_proc)
    return {"job_id": req.job_id, "state": "running", "ptz": ptz_state, "applied": True}


@app.on_event("shutdown")
def shutdown_cleanup() -> None:
    with state_lock:
        jobs = list(ingest_jobs.values())
    for job in jobs:
        _kill_process(job.get("proc"))
