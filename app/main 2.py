from __future__ import annotations

import os
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


class ProbeRequest(BaseModel):
    source_uri_ref: str
    timeout_seconds: float = 6.0


class PreviewStartRequest(BaseModel):
    source_uri_ref: str
    profile: dict = {}


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


app = FastAPI(title="VirtualPTZ Runner Agent", version="0.1.0")
RUNNER_API_KEY = os.getenv("RUNNER_API_KEY", "").strip()
RUNNER_PUBLIC_BASE_URL = os.getenv("RUNNER_PUBLIC_BASE_URL", "http://127.0.0.1:9001").rstrip("/")

preview_jobs: dict[str, dict] = {}
ingest_jobs: dict[str, dict] = {}


def _auth(header: str | None) -> None:
    if not RUNNER_API_KEY:
        return
    if not header or not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = header.removeprefix("Bearer ").strip()
    if token != RUNNER_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid bearer token")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "preview_jobs": len(preview_jobs), "ingest_jobs": len(ingest_jobs)}


@app.post("/runner/probe")
def probe(req: ProbeRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    return {"ok": True, "details": ["runner_probe_stub=ok", f"timeout={req.timeout_seconds}"]}


@app.post("/runner/preview/start")
def preview_start(req: PreviewStartRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    job_id = str(uuid4())
    preview_jobs[job_id] = {"source_uri_ref": req.source_uri_ref, "created_at": datetime.utcnow().isoformat()}
    return {
        "job_id": job_id,
        "preview_url": f"{RUNNER_PUBLIC_BASE_URL}/runner/preview/stream/{job_id}",
    }


@app.get("/runner/preview/stream/{job_id}")
def preview_stream(job_id: str) -> HTTPException:
    raise HTTPException(status_code=501, detail="Implement real MJPEG stream on this runner")


@app.post("/runner/preview/stop")
def preview_stop(req: PreviewStopRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    preview_jobs.pop(req.job_id, None)
    return {"stopped": True}


@app.post("/runner/ingest/start")
def ingest_start(req: IngestStartRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    job_id = str(uuid4())
    ingest_jobs[job_id] = {
        "session_id": req.session_id,
        "state": "running",
        "restart_count": 0,
        "last_error": None,
        "replica_index": req.replica_index,
        "created_at": datetime.utcnow().isoformat(),
    }
    return {"job_id": job_id, "state": "running"}


@app.post("/runner/ingest/stop")
def ingest_stop(req: IngestStopRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    if req.job_id in ingest_jobs:
        ingest_jobs[req.job_id]["state"] = "stopped"
    return {"state": "stopped"}


@app.post("/runner/job/status")
def status(req: JobStatusRequest, authorization: str | None = Header(default=None)) -> dict:
    _auth(authorization)
    job = ingest_jobs.get(req.job_id)
    if not job:
        return {"job_id": req.job_id, "state": "unknown", "restart_count": 0, "last_error": None}
    return {
        "job_id": req.job_id,
        "state": job["state"],
        "restart_count": job["restart_count"],
        "last_error": job["last_error"],
    }
