# HomeBuddy Proactive AI

HomeBuddy Proactive AI is a containerized service for smart glasses. It receives finalized speech-to-text events over WebSocket, follows the conversation, decides whether a short intervention would be useful, retrieves relevant memory, and sends a concise card back to HomeBuddy.

It is designed for a Raspberry Pi 5 and ARM64 Docker, but also runs on Docker Desktop for development.

## Included components

- FastAPI WebSocket API for HomeBuddy/Soniox transcripts
- rolling, time-bounded conversation context
- proactive detector with cooldown and duplicate controls
- Ollama attention gating and insight generation
- durable transcripts, memories, insights, and feedback
- text-file conversation simulator
- Docker Compose deployment and automated tests

The current implementation uses **SQLite, not Redis**. Runtime data persists in a Docker volume.

## Architecture

```text
HomeBuddy / glasses microphone
              │
              ▼
          Soniox STT
              │ finalized transcript
              ▼
       WebSocket /v1/ws
              │
              ▼
      TranscriptBuffer ─────► SQLite transcript history
              │
              ▼
      ProactiveDetector
         │           │
         │ no value  └────────► acknowledge and wait
         ▼
       MemoryEngine ──────────► client + global memories
              │
              ▼
           Ollama
              │
              ▼
        insight event
              │
              ▼
    HomeBuddy ────────────────► smart-glasses display
```

## Requirements

- Raspberry Pi 5, Linux, or Docker Desktop on macOS/Windows
- Docker Engine with Docker Compose v2
- an Ollama server reachable from the API container
- a model already available to Ollama
- TCP port `8080` available for this API
- TCP port `11434` reachable on the Ollama host

Python is not required for Docker operation. Python 3.11+ is only needed for local development.

## 1. Prepare Ollama

List the available models:

```bash
ollama list
```

If necessary, install a model suitable for the machine's memory:

```bash
ollama pull <model-name>
```

The model name in `.env` must exactly match a name shown by `ollama list`.

Ollama must listen on its LAN interface, not only `127.0.0.1`. A foreground example is:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

If Ollama runs as a system service, configure the same `OLLAMA_HOST` value in that service and restart it. Keep port `11434` restricted to the trusted LAN.

Verify connectivity from the Docker host:

```bash
curl http://192.168.68.112:11434/api/tags
```

Replace `192.168.68.112` with the Ollama machine's actual LAN address.

The repository defaults to `qwen3.5:cloud`. That model may use Ollama's cloud service and is not fully offline. For local/offline operation, install a local model and configure its exact name.

## 2. Download and configure

```bash
git clone https://github.com/FilipposPirpilidis/ProactiveAI.git
cd ProactiveAI
cp .env.example .env
```

Edit `.env`:

```dotenv
OLLAMA_BASE_URL=http://192.168.68.112:11434
OLLAMA_MODEL=qwen3.5:cloud
API_TOKEN=
DETECTOR_MODE=conversate
LOG_LEVEL=INFO
```

Use the Ollama host's LAN IP even when Ollama and Docker run on the same Raspberry Pi. Ollama must accept connections from the Docker bridge network.

For a trusted private-LAN test, `API_TOKEN` may remain empty. Before exposing the API to another network, generate a long random token:

```bash
openssl rand -hex 32
```

Put the value in `.env` without quotes.

## 3. Start the API

```bash
docker compose up --build -d proactive-ai
```

Check state and logs:

```bash
docker compose ps
docker compose logs -f proactive-ai
```

Verify the API process:

```bash
curl http://localhost:8080/health
```

```json
{"status":"ok"}
```

Verify the API and Ollama connection:

```bash
curl http://localhost:8080/ready
```

```json
{"status":"ready","model":"qwen3.5:cloud"}
```

`/health` only checks the API process. `/ready` calls Ollama's `/api/tags` and returns HTTP `503` when Ollama is unreachable.

Generated HTTP documentation is available at `http://PI_ADDRESS:8080/docs`.

## 4. Run the simulator

The simulator acts like HomeBuddy forwarding Soniox results. It reads transcript actions from a text file and uses the real WebSocket API.

Run the default file:

```bash
docker compose --profile simulator run --rm simulator
```

Run the included Greek regression conversation:

```bash
docker compose --profile simulator run --rm \
  -e SIMULATOR_TEXT_FILE=/input/real-test-regression-greek.txt \
  -e SIMULATOR_LANGUAGE=el \
  simulator
```

Run a custom file placed in `simulator-input/`:

```bash
docker compose --profile simulator run --rm simulator \
  python scripts/simulator.py file \
  --file /input/my-conversation.txt \
  --language en
```

Run the built-in protocol smoke test:

```bash
docker compose --profile simulator run --rm simulator \
  python scripts/simulator.py scenario
```

Text-file example:

```text
# Comments and blank lines are ignored.
PARTIAL: We should probably
FINAL: We should probably leave for the airport at seven.
WAIT: 1.5
FINAL: What time should we arrive at the airport?
EXPECT_INSIGHT:
FINAL: What time did you say we should arrive?
EXPECT_NO_INSIGHT:
```

| Syntax | Meaning |
|---|---|
| `text` or `FINAL: text` | Send a finalized transcript |
| `PARTIAL: text` | Send interim text; acknowledged but not processed |
| `WAIT: seconds` | Pause for 0–300 seconds |
| `EXPECT_INSIGHT:` | Require an insight for the preceding transcript |
| `EXPECT_INSIGHT: phrase` | Require an insight containing `phrase` |
| `EXPECT_NO_INSIGHT:` | Require no insight for the preceding transcript |

Simulator options include `--url`, `--language`, `--client-id`, `--session-id`, `--token`, and `--timeout`.

## Connect HomeBuddy

Open one WebSocket per active listening session:

```text
ws://PI_ADDRESS:8080/v1/ws?client_id=homebuddy-01&session_id=walk-2026-08-31
```

With authentication enabled:

```text
ws://PI_ADDRESS:8080/v1/ws?client_id=homebuddy-01&session_id=walk-2026-08-31&token=YOUR_TOKEN
```

URL-encode query values. Use `wss://` through a TLS reverse proxy outside a trusted LAN.

- `client_id` identifies the user/device and scopes long-term memory. Keep it stable.
- `session_id` identifies one conversation and scopes transcript context, cooldown state, and recent insight deduplication.
- Use a new `session_id` for a genuinely new conversation. Reusing one restores recent context after reconnection.
- `event_id` identifies a Soniox result. Keep it stable when replaying the same finalized result.

The server first sends:

```json
{
  "type": "ready",
  "client_id": "homebuddy-01",
  "session_id": "walk-2026-08-31",
  "model": "qwen3.5:cloud",
  "detector_mode": "conversate"
}
```

### Transcript event

```json
{
  "type": "transcript",
  "event_id": "soniox-result-42",
  "text": "What should I bring to the appointment tomorrow?",
  "is_final": true,
  "speaker": "owner",
  "language": "en",
  "timestamp": "2026-08-31T09:31:22Z"
}
```

| Field | Required | Description |
|---|---|---|
| `type` | Yes | Must be `transcript` |
| `text` | Yes | Non-empty text, maximum 8,000 characters |
| `is_final` | No | Defaults to `true`; only final text is evaluated and stored |
| `event_id` | No | Defaults to a UUID; a stable Soniox ID is recommended |
| `speaker` | No | Speaker label, maximum 100 characters |
| `language` | No | Language code such as `en` or `el` |
| `timestamp` | No | ISO-8601 timestamp; defaults to current UTC time |

Partial transcripts receive:

```json
{"type":"ack","event_id":"soniox-result-41","processed":false,"reason":"partial"}
```

Final transcripts always receive an acknowledgement. When nothing should be displayed:

```json
{
  "type": "ack",
  "event_id": "soniox-result-42",
  "processed": true,
  "triggered": false,
  "reason": "no_actionable_signal"
}
```

When an insight will follow:

```json
{
  "type": "ack",
  "event_id": "soniox-result-42",
  "processed": true,
  "triggered": true,
  "reason": "strong_local_signal"
}
```

The following event is the glasses card:

```json
{
  "type": "insight",
  "insight_id": "29c025a8-6b64-4572-9568-ccfdc3f875d4",
  "text": "Bring your ID and the signed form.",
  "intent": "question",
  "confidence": 0.9,
  "created_at": "2026-08-31T09:31:23.120000+00:00"
}
```

Client flow:

1. Wait for `ready` before sending transcripts.
2. Send events in chronological order.
3. For `ack.triggered: false`, continue listening.
4. For `ack.triggered: true`, wait for the next `insight` or `error`.
5. Render `insight.text` and retain `insight_id` for feedback.
6. Reconnect with backoff after failure, reusing the session only when continuing the same conversation.

### Ping and feedback

```json
{"type":"ping"}
```

```json
{"type":"pong"}
```

```json
{
  "type": "feedback",
  "insight_id": "29c025a8-6b64-4572-9568-ccfdc3f875d4",
  "useful": true
}
```

```json
{
  "type": "feedback_saved",
  "insight_id": "29c025a8-6b64-4572-9568-ccfdc3f875d4"
}
```

WebSocket error codes include `invalid_message`, `unsupported_type`, and `llm_unavailable`. An `llm_unavailable` event has `retryable: true`.

## Add memories

Reminder/task utterances that trigger are captured automatically. A trusted service can also add memory over HTTP:

```bash
curl -X POST http://localhost:8080/v1/memories \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{
    "client_id": "homebuddy-01",
    "kind": "preference",
    "content": "Alex prefers short visual answers"
  }'
```

When `API_TOKEN` is empty, omit the authorization header. Use `client_id: "global"` only for facts that should be visible to every client. There is currently no public endpoint for listing or deleting memories.

Retrieval is lightweight: the newest 100 client/global memories are ranked using Unicode lexical overlap, and up to `MEMORY_RESULT_LIMIT` entries are supplied to the detector.

## Detector modes

| Mode | Behavior |
|---|---|
| `conversate` | Default. Evaluates every meaningful final utterance with Ollama; best for continuous assistance. |
| `hybrid` | Calls Ollama only for ambiguous utterances with some local signal. |
| `heuristic` | Uses local detection rules only; fastest but least conversational. |

The detector can surface questions, context, corrections, definitions, suggestions, warnings, reminders, tasks, and decisions. It suppresses filler speech, credential-like phrases, duplicate utterances, repeated cards, low-priority cooldown events, and subjective human-to-human questions that do not merit interruption.

Questions and high-priority signals may bypass cooldown. Only the latest utterance can trigger; older speech is supporting context.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://192.168.68.112:11434` | Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `qwen3.5:cloud` | Exact Ollama model name |
| `OLLAMA_TIMEOUT_SECONDS` | `45` | Model request timeout |
| `API_TOKEN` | empty | Optional HTTP bearer/WebSocket query token |
| `DETECTOR_MODE` | `conversate` | `conversate`, `hybrid`, or `heuristic` |
| `DETECTOR_THRESHOLD` | `0.62` | Minimum accepted trigger confidence |
| `INSIGHT_COOLDOWN_SECONDS` | `20` | Low-priority per-session cooldown |
| `TRANSCRIPT_WINDOW_SECONDS` | `90` | Rolling context age |
| `TRANSCRIPT_MAX_ITEMS` | `40` | Rolling context item limit |
| `MEMORY_RESULT_LIMIT` | `5` | Maximum retrieved memories |
| `DATABASE_PATH` | `/data/homebuddy.db` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Python log level |

Compose forwards `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `API_TOKEN`, `DETECTOR_MODE`, and `LOG_LEVEL` from `.env`, and sets `DATABASE_PATH` to the volume. Other tuning variables use their defaults. To override one in Docker, add it under `services.proactive-ai.environment` in `compose.yaml`, then recreate the service:

```bash
docker compose up -d --force-recreate proactive-ai
```

## Persistence and privacy

SQLite lives at `/data/homebuddy.db` in the `homebuddy-data` volume. It contains:

- finalized transcripts;
- explicit and automatically captured memories;
- generated insights;
- usefulness feedback.

Partial transcripts are not persisted. Final transcripts are stored before proactive filtering, including speech later classified as noise or sensitive. Do not send speech unless the user has consented to storage.

The in-memory buffer defaults to 90 seconds and 40 utterances. SQLite records remain until deliberately removed.

Stop without deleting data:

```bash
docker compose down
```

Restart:

```bash
docker compose up -d proactive-ai
```

Do not run `docker compose down -v` unless you intentionally want to delete the database volume.

## Security

- Set `API_TOKEN` outside a trusted development LAN.
- Put HTTPS/WSS termination in front of port `8080` before internet exposure.
- Never expose Ollama port `11434` publicly.
- Restrict firewall access to trusted clients.
- Treat transcripts and memories as private user data.
- Protect `.env`; it is ignored by Git.
- Rotate tokens that appear in logs or URLs.

The container runs as an unprivileged `homebuddy` user, drops Linux capabilities, enables `no-new-privileges`, and writes state only to `/data`.

## Common operations

```bash
# Update and rebuild
git pull
docker compose up --build -d proactive-ai

# Restart
docker compose restart proactive-ai

# Recent logs
docker compose logs --tail=200 proactive-ai

# Stop without deleting data
docker compose down
```

## Troubleshooting

### `/ready` returns 503

```bash
curl http://OLLAMA_IP:11434/api/tags
```

Confirm Ollama is running, bound to `0.0.0.0:11434`, allowed through the firewall, and configured with the correct LAN IP. If tags work but chat fails, ensure `OLLAMA_MODEL` exactly matches an available model.

### The container is unhealthy

```bash
docker compose ps
docker compose logs --tail=200 proactive-ai
```

Check that port `8080` is free and the persistent volume is writable.

### HomeBuddy receives no cards

- Ensure Soniox final events use `"is_final": true`.
- Ensure messages use valid JSON and `"type": "transcript"`.
- Inspect `ack.reason`; silence may be intentional.
- Use `DETECTOR_MODE=conversate` for continuous assistance.
- Check Ollama latency and service logs.
- Use a new `session_id` for a new conversation.

### Responses are slow

Latency is usually Ollama inference. Use a smaller local model, reduce model contention, prefer wired networking, or use `hybrid`/`heuristic`. Explicit reminder/task cards are deterministic and do not need a generation call after detection.

### Authentication fails

HTTP writes require `Authorization: Bearer YOUR_TOKEN`. WebSocket connections use `?token=YOUR_TOKEN`. Restart the container after changing `.env`.

## Local development and tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
docker compose config
docker compose build
```

Run without Docker only for development:

```bash
DATABASE_PATH=./homebuddy.db \
OLLAMA_BASE_URL=http://192.168.68.112:11434 \
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Use Docker Compose for Raspberry Pi operation so dependencies remain reproducible.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
