# Realtime speech-to-speech (`/v1/realtime`)

A WebSocket voice agent that speaks a subset of the **OpenAI Realtime** protocol
(GA event names; `?events=beta` for the older ones). OpenAI SDKs, pipecat or
LiveKit clients connect unchanged. Browser demo: `GET /realtime`, **off by
default** (`zallama realtime demo on|off`, then restart; while on, the page is
public but its WebSocket still needs the API key; off, `/realtime` is a 404).

```
mic PCM16 24 kHz ─▶ Silero VAD ─▶ ASR ─▶ LLM (streamed, tools) ─▶ phrase chunker ─▶ TTS (streamed) ─▶ PCM16 24 kHz
```

Code: `server/routes/realtime.py` (session, protocol, pipeline),
`server/realtime_vad.py` (Silero VAD), `server/routes/realtime_demo.html`.

## Setup

```yaml
# ~/.zallama/config.yaml
realtime:
  llm_model: "gemma-e2b-voice"      # Gemma-4-E2B, ctx_size 8192, parallel 1, evict_group voice
  asr_model: "parakeet-tdt-v3-cpu"  # CPU, no VRAM
  tts_model: "pocket-tts"           # language-routing entry → pocket-tts-fr / -en
  voice: "estelle"
```

The VAD needs `onnxruntime` (in requirements.txt) and `silero_vad.onnx` in
`models_dir`:

```bash
/opt/scripts/wget.sh https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx /bank2/zallama/models
```

All tunables, with their defaults, are in `config/config.example.yaml` (`realtime:`).
`silence_duration_ms` (500) is the main latency/false-cut trade-off.

## How latency is won

| Trick | Effect |
|---|---|
| **Speculative turns** | At 200 ms of silence, ASR and the LLM already start; nothing is sent until the silence reaches 500 ms and the turn commits. If the user speaks again, the turn is dropped. ASR and most of the LLM's time to first token hide in the endpointing wait. |
| **Early first phrase** | The first phrase goes to the TTS at the first comma after 3 words (or 8 words); then whole sentences. |
| **Streamed TTS** | pocket-tts `/stream` returns PCM frames as they decode: first audio 25–60 ms after a phrase. kokoro (CPU, not streamed) took ~1 s per phrase. |
| **Rules-only normalization** | pocket-tts's built-in tn rules; the Gemma LLM normalizer pass (~190 ms/sentence) is skipped. |
| **Warm prompt** | Each `session.update` prefills system prompt + tools (`max_tokens: 1`) so the first turn hits the llama-server prompt cache. |
| **Held models** | The session holds its LLM/ASR/TTS instances (in-flight), so nothing evicts them between turns. |

## Measured (2026-10-02, RTX 4090, French)

Simulated mic (`pocket-tts` voice "jean" played in real time), from the real
end of speech to the first audio byte sent:

| LLM | VRAM | ASR | LLM TTFT | first audio | tools (4 cases) |
|---|---|---|---|---|---|
| **gemma-e2b-voice** (Gemma-4-E2B QAT, ctx 8192) | 1.8 GB | 78–168 ms | 24–36 ms | **506–531 ms** | 30/30 |
| gemma-4-12b-it-Q4_K_M (MTP head, 60–100 % accepted) | 11.8 GB | 73–139 ms | 79–160 ms | 506–641 ms | 4/4 |
| Qwen3.8-27B-Q4_K_M (MTP, hybrid) | 20.8 GB | 75–135 ms | 240–380 ms | 640–830 ms | 4/4 |
| qwen3-0.6b-q8_0 | 1.4 GB | 80–170 ms | 16–20 ms | 506 ms | invents answers |

506 ms is the floor set by `silence_duration_ms: 500`: with Gemma 12B, ASR + LLM
+ first phrase + TTS all fit inside the speculative window most of the time.
**gemma-e2b-voice is the configured voice LLM**: same speed floor, and it runs
*beside* the resident 27B (evict_group `voice`): 27B + E2B + pocket-tts =
23.95 GB of 24.56, so ~600 MB margin; loading the embedding model too may not fit.
After a tool result it starts speaking 110–140 ms after `response.create`.
Gemma 12B is smarter in open conversation but can't share the card with the 27B
(a voice session would evict it).

The tools column for E2B is 10 single-step French requests × 3 at temperature
0.7 (weather, timer incl. "un quart d'heure", SMS, lights, time, two no-tool
questions). **The system prompt decides it**: with a longer speech-style prompt
("natural speech, no lists… call the tool right away instead of saying you
will"), E2B narrated the action instead of calling the tool: 6–7/30. A
French persona line ("Tu t'appelles Zal") cost a few more. The shipped
`STYLE_PROMPT` (two sentences, ending "Use the tools to act or to get facts.")
gets 30/30; the demo's instructions on top, 29/30. Re-run the test when
changing it. Qwen3.6-35B-A3B
and gemma-4-26B-A4B weren't testable: their GGUFs (`/volume/models/unsloth/`)
are gone.

- Tool call (27B): `get_weather({"city":"Lyon"})` emitted ~470 ms after commit;
  after the client's `function_call_output` + `response.create`, the spoken
  answer started 345 ms later.
- Barge-in: the response is cancelled ~160 ms after the user starts speaking
  over it (`barge_in_ms: 250` of speech, counted from the VAD start).
- qwen3-0.6b answers "Quel temps fait-il à Lyon ?" with an invented forecast
  instead of calling `get_weather`: too small for tool use, as in
  [qwen3-0.6b-agentic.md](qwen3-0.6b-agentic.md).

### The 27B's fixed ~200 ms per request: context checkpoints

Qwen3.8-27B is a hybrid (recurrent + attention) model. With llama-server's
context checkpoints on (default 32), every request pays ~200–280 ms of prompt
eval even for 17 new tokens. Measured on a scratch copy of the same launch line:

| flags | append-only turn TTFT | turn after a cancelled/diverged request |
|---|---|---|
| default (checkpoints 32) | 250–340 ms | 240 ms (rolls back, 20 tokens) |
| `--ctx-checkpoints 2` | same | same |
| `--ctx-checkpoints 0` | **52 ms** | 530 ms: full re-process of the 1.3k-token history (grows with it) |
| `--no-cache-idle-slots`, no MTP, ctx 8192 | no change | — |

Without checkpoints a hybrid model cannot roll back, and realtime diverges
often (every dropped speculative turn, every barge-in truncation), so the
production 27B keeps its checkpoints. A dedicated voice LLM instance with
`--ctx-checkpoints 0` and `speculative_ms` disabled (set it ≥
`silence_duration_ms`) would trade the speculation for a ~50 ms TTFT.

## Protocol notes

- Audio in: `input_audio_buffer.append`, base64 PCM16 mono, 24 kHz by default
  (`session.audio.input.format.rate` for 16/48 kHz). Audio out: PCM16 24 kHz.
- Tools run on the client: the response ends with a `function_call` item
  (`response.function_call_arguments.done`); send `conversation.item.create`
  with a `function_call_output`, then `response.create`.
- `turn_detection: null` = manual: `input_audio_buffer.commit`, then `response.create`.
- `output_modalities: ["text"]` skips the TTS (`response.output_text.delta`).
- Barge-in history: the assistant message is cut to what was played. The
  server estimates it from wall-clock time since the first audio byte; a client
  `conversation.item.truncate {audio_end_ms}` overrides it.
- `?model=<text model>` picks another LLM for the session; extensions in
  `session.update`: `language`, `turn_detection.speculative_ms`,
  `turn_detection.barge_in_ms`. `response.done.response.zallama.latency_ms`
  reports `asr_ms`, `llm_ttft_ms`, `tts_first_ms`, `first_audio_ms`.
- Auth: same API key as HTTP (the HTTP middleware never sees WebSockets), as
  `Authorization: Bearer` or the `openai-insecure-api-key.<key>` subprotocol
  (browsers). Loopback is let through.
- Echo: clients must cancel echo (browser `echoCancellation: true`), or the
  assistant's own voice triggers barge-in.
- Browser mic needs a secure context: the demo works on `http://localhost`
  or over HTTPS, not on plain-HTTP LAN addresses.
