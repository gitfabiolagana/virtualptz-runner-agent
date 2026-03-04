# VirtualPTZ Runner Agent (Linux)

Agent minimale da installare sulle macchine Linux runner.

## Avvio

```bash
cd /Users/fabiolagana/Documents/codex_projects/virtualptz-runner-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
RUNNER_API_KEY=tokenA RUNNER_PUBLIC_BASE_URL=http://10.0.1.10:9001 uvicorn app.main:app --host 0.0.0.0 --port 9001
```

Swagger/OpenAPI:

- Swagger UI: `http://127.0.0.1:9001/docs`
- ReDoc: `http://127.0.0.1:9001/redoc`
- OpenAPI JSON: `http://127.0.0.1:9001/openapi.json`

## Config utili

- `RUNNER_API_KEY` token richiesto dal gateway
- `RUNNER_PUBLIC_BASE_URL` URL pubblico del runner per stream preview
- `RUNNER_INGEST_OUTPUT_URI` opzionale; se vuoto l'ingest va su `null`, se valorizzato pubblica su destinazione (es. `rtmp://...`)
- `RUNNER_PTZ_MIN_APPLY_INTERVAL_SECONDS` (default `0.35`) cooldown anti-jitter PTZ
- `RUNNER_PTZ_MIN_DELTA_XY` (default `0.008`) delta minimo center X/Y per riapplicare crop
- `RUNNER_PTZ_MIN_DELTA_ZOOM` (default `0.015`) delta minimo zoom per riapplicare crop
- `RUNNER_PTZ_HOT_UPDATE_ENABLED` (default `1`) abilita PTZ hot-update via `ffmpeg zmq` (se disponibile)
- `RUNNER_PTZ_HOT_UPDATE_TIMEOUT_MS` (default `350`) timeout comando ZMQ
- `RUNNER_PTZ_RESTART_MIN_INTERVAL_SECONDS` (default `1.15`) fallback restart throttled quando hot-update fallisce
- `RUNNER_PTZ_WORKER_POLL_SECONDS` (default `0.08`) polling worker PTZ

WebRTC preview configuration
- `RUNNER_WEBRTC_ENABLED` (0/1) abilita il flusso WebRTC preview (default 0)
- `RUNNER_WEBRTC_PUBLISH_BASE_URI` base URI per la destinazione di publish (es. RTSP o RTMP). If this
  value starts with `rtmp://` the runner will publish using FLV (`-f flv`) so media servers like SRS can
  ingest via RTMP. If it is an RTSP URI the runner will publish via RTSP.
- `RUNNER_WEBRTC_WHEP_BASE_URL` base URL used to build the runner preview endpoint. For some media servers
  (e.g. SRS) a WHIP-style `whip-play` endpoint expects query parameters; the runner will build a compatible
  preview URL when `whip-play` appears in this value.

## Stato attuale

L'agent avvia processi FFmpeg reali per preview e ingest, espone stato job e stop con terminate/kill.

PTZ update:
- percorso primario: hot-update runtime su filtro `crop` tramite `zmq` (senza restart ffmpeg)
- fallback automatico: restart ingest con throttling se ZMQ non disponibile o comando fallisce

Endpoint principali:

- `POST /runner/probe`
- `POST /runner/preview/start`
- `GET /runner/preview/stream/{job_id}`
- `POST /runner/preview/stop`
- `POST /runner/ingest/start`
- `POST /runner/ingest/stop`
- `POST /runner/job/status`
- `POST /runner/tracking/status`
- `POST /runner/tracking/update` (push metriche da detector esterno)

## Nota preview file (resume)

`POST /runner/preview/start` supporta `profile.start_seconds` per avviare la preview file da un offset (utile per mantenere la posizione player quando cambi vista).

## Tracking Worker (push automatico)

Script: `scripts/tracking_push_worker.py`

Esempio simulato:

```bash
python scripts/tracking_push_worker.py \
  --runner-base-url http://127.0.0.1:9001 \
  --job-id <INGEST_JOB_ID> \
  --api-key <RUNNER_API_KEY> \
  --mode sim \
  --interval 1.0
```

Esempio con file JSON del detector (aggiornato in loop):

```bash
python scripts/tracking_push_worker.py \
  --runner-base-url http://127.0.0.1:9001 \
  --job-id <INGEST_JOB_ID> \
  --api-key <RUNNER_API_KEY> \
  --mode file \
  --input-file /tmp/tracking.json \
  --interval 0.5
```

Formato JSON input (`/tmp/tracking.json`):

```json
{
  "tracking_confidence": 0.83,
  "tracking_state": "stable",
  "fallback_active": false,
  "occlusion_level": 0.12,
  "source": "yolo-bytetrack"
}
```

## Launcher ingest + tracking (template)

Script: `scripts/start_ingest_with_tracking.py`

Avvia ingest, cattura il `job_id` e lancia il worker tracking sullo stesso job.

```bash
python scripts/start_ingest_with_tracking.py \
  --runner-base-url http://127.0.0.1:9001 \
  --api-key <RUNNER_API_KEY> \
  --session-id session-basket-001 \
  --source-uri-ref rtmp://streaming.pncbasket.it:1935/live/stream-0 \
  --profile balanced \
  --tracking-mode sim \
  --tracking-interval 1.0
```

Per collegare detector reale:

```bash
python scripts/start_ingest_with_tracking.py \
  --runner-base-url http://127.0.0.1:9001 \
  --api-key <RUNNER_API_KEY> \
  --session-id session-basket-001 \
  --source-uri-ref rtmp://streaming.pncbasket.it:1935/live/stream-0 \
  --profile balanced \
  --tracking-mode file \
  --tracking-input-file /tmp/tracking.json \
  --tracking-interval 0.5
```
