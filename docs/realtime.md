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

Registry names of the LLM entries read `<model>-<quant>-<ctx>k<slots>s-<options>-<kv>-<vram>g`:
options `m` MTP, `d` dflash, `v` mmproj on GPU, `vc` mmproj on CPU, `t` thinking
(omitted when none); kv `k4`/`k8`/`k16`; vram is the measured `mem_gb`. Older
names stay as aliases. E.g. `qwen27b-q4-96k2s-mv-k4-20.8g` = Qwen3.8-27B Q4, 96K
context, 2 slots, MTP, vision on GPU, q4_0 KV, 20.8 GB.

```yaml
# ~/.zallama/config.yaml
realtime:
  llm_model: "gemma-e2b-q4-32k1s-k8-1.9g"   # alias: gemma-e2b-voice
  asr_model: "parakeet-tdt-v3-cpu"  # CPU, no VRAM
  tts_model: "pocket-tts-cpu"       # CPU language router → pocket-tts-fr-cpu / -en-cpu (see VRAM below)
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
| **gemma-e2b-q4-32k1s-k8-1.9g** (alias gemma-e2b-voice) | 1.9 GB | 78–168 ms | 24–36 ms | **506–531 ms** | 30/30 |
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

### Context size and history

Gemma-4-E2B's KV cache is tiny (20 of 35 layers share KV, one KV head, most
layers on a 512-token sliding window), so context is cheap. Measured VRAM of
gemma-e2b-voice (q8_0 KV): 8K 1.84 GB, **32K 1.92 GB**, 128K 2.51 GB. 32K is
the largest that fits beside the 27B + pocket-tts (~24.1 of 24.56 GB).

The session keeps the conversation under the LLM's context: past
`history_tokens` (default: the slot's ctx_size − max_tokens − 256, tokens
estimated at 3 chars each), the oldest turns are dropped down to 60 % of the
budget in one go, starting again on a user message. Each trim re-prefills the
history once, hence the big step instead of one turn at a time.

### VRAM: run the voice TTS on CPU next to a big resident model

pocket-tts allocates its working memory *during* each synthesis (llama-server
reserves everything at startup). With the 27B (21.3 GB) + gemma-e2b-voice
32K (1.9 GB) + pocket-tts on CUDA (0.8 GB), the card sat at 24 072 of
24 564 MiB and pocket-tts failed every request with `CUDA_ERROR_OUT_OF_MEMORY`.

So the voice uses CPU entries: `pocket-tts-cpu` (language router, alias `pocket-tts-voice`) →
`pocket-tts-fr-cpu` / `pocket-tts-en-cpu` (`device: cpu`, `threads: 4`,
`mem_gb: 0.01`, evict_group `voice`). Measured: first audio 60–130 ms per
phrase (GPU: 25–60 ms), 4.6–6.4× realtime, so streaming stays ahead of
playback; peak VRAM during a full fr/en session 23 281 MiB (1.28 GB free).
Switching language costs one cold model load (~0.8 s) the first time.

```yaml
realtime:
  tts_model: "pocket-tts-cpu"
```

### Making room: a lighter entry for the resident 27B

What the Qwen3.8-27B-Q4_K_M launch line costs in VRAM (measured, q4_0 KV, MTP),
and the time to answer about a 1280×800 screenshot (~1 000 image tokens):

| variant | VRAM | screenshot |
|---|---|---|
| parallel 2 + kv_unified, ctx 98304, mmproj on GPU (`qwen27b-q4-96k2s-mv-k4-20.8g`, alias `Qwen3.8-27B-Q4_K_M`) | 21 298 MiB | |
| parallel 2, no kv_unified | 21 232 MiB | |
| parallel 1 | 20 700 MiB | 1.2 s |
| **parallel 1, ctx 65536** (`qwen27b-q4-64k1s-mv-k4-19.4g` (alias `Qwen3.8-27B-Q4_K_M-lite`)) | **19 876 MiB** | **1.2 s** |
| parallel 1, ctx 32768 | ~17 800 MiB | |
| parallel 1 + `no_mmproj_offload: true` (mmproj on CPU), ctx 98304 | 19 562 MiB | 11.7 s |

kv_unified costs nothing by itself (the slots share one ctx_size); the second
slot costs 0.6 GB, the BF16 mmproj on GPU 1.1 GB, and every 32K of context
~0.9 GB. The mmproj stays on GPU: on CPU a screenshot takes 11.7 s, too slow
for a browser agent. 64K still holds ~50 screenshot steps. With the 64K one-slot entry, the
card holds the 27B + gemma-e2b-voice + pocket-tts on CUDA at 22.6 GB (1.9 GB
free). Two entries on one file are two processes: a client asking for the
other name evicts this one and reloads it.

### KV cache type: same type for K and V

Measured on Gemma 4 12B, 98K context, 1 slot, MTP, vision on GPU (VRAM after
a vision request; prefill on a 4.3K-token prompt):

| cache_type_k / v | VRAM | prefill |
|---|---|---|
| f16 / f16 | 10 866 MiB | 4 674 tok/s |
| q8_0 / q8_0 | 10 220 MiB | 5 080 tok/s |
| **q4_0 / q4_0** | **9 716 MiB** | **5 551 tok/s** |
| q8_0 / q4_0 | 9 466 MiB | **94 tok/s** |

Mixed K/V types look smallest but have no CUDA flash-attention kernel: the
attention falls back to the CPU (a 30K-token prompt timed out after 10 min).
q4_0/q4_0 kept quality here: needle-in-a-haystack 3/3 at 31.8K tokens, tool
calls 30/30. Entry: `gemma12b-q4-96k1s-mv-k4-9.5g`.

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
- Tool calls written as text: when llama.cpp's parser misses a call, the
  content carries the model's raw syntax (Gemma `call:name{city:<|"|>Lyon<|"|>}`,
  inside `<|tool_call>…<tool_call|>`; Hermes/Qwen `<tool_call>{"name":…}</tool_call>`).
  The session holds back text that may start such a call (even split across
  chunks), runs a complete one naming a declared tool as a real
  `function_call`, and never sends it to the TTS. A `call:` naming no declared
  tool is spoken as normal text; prose after a text call (usually an invented
  tool result) is dropped. Each one is logged ("tool call written as text").
- Auth: same API key as HTTP (the HTTP middleware never sees WebSockets), as
  `Authorization: Bearer` or the `openai-insecure-api-key.<key>` subprotocol
  (browsers). Loopback is let through.
- Echo: clients must cancel echo (browser `echoCancellation: true`), or the
  assistant's own voice triggers barge-in.
- Browser mic needs a secure context: the demo works on `http://localhost`
  or over HTTPS, not on plain-HTTP LAN addresses.
