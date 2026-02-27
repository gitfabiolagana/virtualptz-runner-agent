from __future__ import annotations

import os
import subprocess
import threading
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


app = FastAPI(title="VirtualPTZ Runner Agent", version="0.2.0")
RUNNER_API_KEY = os.getenv("RUNNER_API_KEY", "").strip()
RUNNER_PUBLIC_BASE_URL = os.getenv("RUNNER_PUBLIC_BASE_URL", "http://127.0.0.1:9001").rstrip("/")
INGEST_OUTPUT_URI = os.getenv("RUNNER_INGEST_OUTPUT_URI", "").strip()

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


def _ffmpeg_preview_cmd(source_uri: str, fps: int = 12, scale: str = "720:-1") -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if source_uri.startswith("rtsp://"):
        cmd.extend(["-rtsp_transport", "tcp"])
    elif source_uri.startswith("rtmp://"):
        cmd.extend(["-rw_timeout", "15000000"])
    cmd.extend(["-i", source_uri, "-an", "-vf", f"fps={fps},scale={scale}", "-q:v", "6", "-f", "mjpeg", "pipe:1"])
    return cmd


def _ffmpeg_ingest_cmd(source_uri: str, profile: str) -> list[str]:
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
    cmd.extend(["-i", source_uri, "-an", "-c:v", "libx264", *preset])

    if INGEST_OUTPUT_URI:
        cmd.extend(["-f", "flv", INGEST_OUTPUT_URI])
    else:
        cmd.extend(["-f", "null", "-"])
    return cmd


def _start_ingest_watcher(job_id: str, proc: subprocess.Popen) -> None:
    def _watch() -> None:
        rc = proc.wait()
        with state_lock:
            job = ingest_jobs.get(job_id)
            if not job:
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
    fps = int(req.profile.get("fps", 12)) if isinstance(req.profile, dict) else 12
    scale = str(req.profile.get("scale", "720:-1")) if isinstance(req.profile, dict) else "720:-1"
    cmd = _ffmpeg_preview_cmd(req.source_uri_ref, fps=fps, scale=scale)
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
    cmd = _ffmpeg_ingest_cmd(req.source_uri_ref, req.profile)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError:
        raise HTTPException(status_code=500, detail="failed to start ffmpeg") from None

    with state_lock:
        ingest_jobs[job_id] = {
            "job_id": job_id,
            "session_id": req.session_id,
            "replica_index": req.replica_index,
            "state": "running",
            "restart_count": 0,
            "last_error": None,
            "pid": proc.pid,
            "proc": proc,
            "masked_source": _mask_sensitive_uri(req.source_uri_ref),
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


@app.on_event("shutdown")
def shutdown_cleanup() -> None:
    with state_lock:
        jobs = list(ingest_jobs.values())
    for job in jobs:
        _kill_process(job.get("proc"))
