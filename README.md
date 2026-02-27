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

## Config utili

- `RUNNER_API_KEY` token richiesto dal gateway
- `RUNNER_PUBLIC_BASE_URL` URL pubblico del runner per stream preview
- `RUNNER_INGEST_OUTPUT_URI` opzionale; se vuoto l'ingest va su `null`, se valorizzato pubblica su destinazione (es. `rtmp://...`)

## Stato attuale

L'agent avvia processi FFmpeg reali per preview e ingest, espone stato job e stop con terminate/kill.
