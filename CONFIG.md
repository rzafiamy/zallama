# Registry parameter reference

Every entry in `registry.yaml` (`<models_dir>/registry.yaml`, e.g.
`/bank2/zallama/models/registry.yaml`) looks like:

```yaml
models:
  - name: my-model
    file: my-model.gguf
    modality: text        # text | embedding | rerank | asr | tts | image (default: text)
    backend: llama-server  # optional, inferred from modality
    artifacts: {}          # optional, e.g. mmproj
    mem_gb: 12.0            # optional, declared VRAM cost for the LRU budget
    pinned: false           # optional, pre-load at startup, exempt from eviction
    evict_group: ""          # optional, overrides the modality-based default eviction group
    aliases: []              # optional, alternate model names
    description: ""          # optional
    params: {}                # backend-specific — see tables below
```

`params` is a free-form dict; only the keys a given **backend** knows how to
translate get forwarded to the subprocess (unrecognized keys are silently
ignored — there is no validation). Which backend an entry uses is picked from
`modality` unless `backend` overrides it:

| modality    | default backend    | endpoint(s)                                      |
|-------------|---------------------|---------------------------------------------------|
| `text`      | `llama-server`      | `/v1/chat/completions`, `/v1/completions`          |
| `embedding` | `embedding-server`  | `/v1/embeddings`                                   |
| `rerank`    | `rerank-server`     | `/v1/rerank`                                       |
| `asr`       | `parakeet-server`   | `/v1/audio/transcriptions`                         |
| `tts`       | `kokoro-server`     | `/v1/audio/speech`                                 |
| `image`     | `sd-server`         | `/v1/images/generations`                           |

## Priority / merge order

`params` are merged with `llama_server.default_params` from `config.yaml`
before being turned into CLI flags: **registry `params` override config
`default_params`**, key by key (`{**default_params, **model_params}`).

For the handful of params that are *also* accepted per-request in the JSON
body (chat sampling knobs, kokoro `voice`/`speed`, sd-server `steps` /
`cfg_scale` / `sampler` / `negative_prompt`), the effective priority is:

```
request body  >  registry params  >  config default_params  >  backend's own built-in default
```

For `llama-server` this happens for free: the CLI flag just becomes the
server's own default, and `llama-server`'s OpenAI route already only falls
back to it when the request omits the field. For `kokoro-server` and
`sd-server`, whose CLI has no such per-field defaulting, the proxy in
`server/routes/openai.py` applies the registry value into the request body
itself when the client didn't set it.

## `evict_group` — scoping LRU eviction

When a model needs to load and capacity (`max_loaded_models` /
`mem_budget_gb`) is tight, Zallama evicts the least-recently-used *non-pinned*
instance that shares the incoming model's **eviction group** — never one
outside it. Every modality has a built-in default group, so this works with
no config at all:

| modality | default `evict_group` |
|---|---|
| `text`, `image` | `primary` |
| `asr`, `embedding`, `rerank`, `tts` | `services` |

In other words: a large, slow-to-reload text or image model only ever gets
evicted to make room for *another* text/image model — never bumped just
because a small ASR or embedding request came in. Conversely, ASR and
embedding trade a single shared slot back and forth between themselves
without ever reaching into the text/image model's slot.

Set `evict_group` explicitly on an entry to override the default — put a
specific model in its own group, opt it out of grouping entirely (an empty
string falls through to "no restriction," the old any-non-pinned-victim
behavior), or invent an unrelated group name:

```yaml
- name: tdt-0.6b-v3-q8_0        # asr — no evict_group needed, defaults to "services"
- name: Qwen3-Embedding-0.6B-Q8_0  # embedding — defaults to "services"
- name: Qwen3.8-27B-Q4_K_M      # text — defaults to "primary"
- name: flux:klein              # image — defaults to "primary"
```

Or set it from the CLI instead of hand-editing the YAML:

```bash
zallama set my-reranker evict_group=custom-slot   # put it in its own group
zallama set my-reranker evict_group=              # opt out of grouping entirely
zallama set my-reranker evict_group=none           # reset to the modality default
```

If capacity is tight and no same-group victim is loaded, Zallama does **not**
fall back to evicting across groups — it logs a warning and admits the
incoming model over budget instead, the same fallback used when every loaded
model is pinned. See
[docs/vram-planning.md](docs/vram-planning.md) for a full worked example.

### `evict_group_mem_budgets` — capping a group's own memory, independent of global slack

`max_loaded_models` and `mem_budget_gb` only pressure eviction once the
**global** count/budget is exceeded. That leaves a gap: if those globals were
sized assuming, say, two `primary` models resident at once, then whenever only
one is actually loaded there's global slack left over — enough for *several*
`services` models (ASR, embedding, autocomplete, ...) to load side by side
without any of them ever pressuring the others out, even though they share an
`evict_group`. Sharing a group only says *who's allowed to evict whom* — it
doesn't by itself cap *how much* of them can be resident together.

`config.yaml`'s `llama_server.evict_group_mem_budgets` closes that gap with a
per-group memory budget that's checked regardless of global slack:

```yaml
llama_server:
  evict_group_mem_budgets:
    services: 1.5   # ASR + embedding + autocomplete share 1.5 GB between them
```

This is a **memory** cap, not an instance-count cap — deliberately: capping
the *count* would block a second small model from loading even when VRAM is
sitting idle, which isn't the goal. With a budget instead, two small models
are free to coexist for as long as their combined `mem_gb` fits inside it;
eviction of the group's LRU member only fires once a new arrival would
actually push the group's own total over its own budget — even if
`max_loaded_models` and `mem_budget_gb` are nowhere near their limits — while
`primary` stays completely unaffected (group-scoped eviction still never
crosses into it). A group with no entry here is uncapped except by the two
global knobs, same as before this option existed. As with the global budget,
it's only meaningful once the group's members carry *measured* `mem_gb`
values — see [Why the default cost estimate is
wrong](docs/vram-planning.md#why-the-default-cost-estimate-is-wrong).

---

### `evict_drain_timeout` — don't kill a backend mid-response

When two models don't fit together and clients alternate between them, every
request evicts the other model and respawns its backend. On its own that's just
slow. The sharp edge is a request landing for model B *while model A is still
streaming a response* — eviction would kill A's backend out from under the
in-flight proxy, and A's client gets a truncated stream / `502`.

Zallama tracks in-flight requests per backend. Eviction prefers a victim with
nothing in flight; when the only eligible victim is busy, it waits up to
`evict_drain_timeout` seconds for that backend to go idle before killing it:

```yaml
llama_server:
  evict_drain_timeout: 30   # seconds; 0 = evict immediately (old behavior)
```

The wait is bounded — if requests keep the backend busy past the timeout it's
evicted anyway, since an unbounded stall is worse. Idle-sweep (`idle_timeout`)
also skips a backend that's still serving, so a generation longer than
`idle_timeout` won't be cut off. None of this makes two oversized models
coexist — the real fix for alternation thrash is to route traffic to **one**
model, or give it enough `mem_budget_gb` headroom that both stay resident.

---

## `text` (backend: `llama-server`)

Also covers vision (add an `mmproj` artifact) and any entry with
`backend: llama-server` explicitly. Source: `LlamaServerBackend` in
`server/backends.py`.

### Artifacts

| key | CLI flag | notes |
|---|---|---|
| `mmproj` | `--mmproj` | vision projector |
| `draft` | `--model-draft` | standalone draft model for `spec_type: draft-simple` (and friends). Not needed for `draft-mtp` — that head lives inside the main GGUF; check the main model's tensor list for a `blk.N.nextn.*` block before assuming a same-repo `*-mtp-*.gguf`/`*-draft-*.gguf` sibling is meant to be wired here — some repos ship it as a leftover/back-compat copy of the already-embedded head rather than a genuinely separate model. |

### Value params

| key | CLI flag | notes |
|---|---|---|
| `ctx_size` | `--ctx-size` | context length in tokens |
| `n_gpu_layers` | `--n-gpu-layers` | `99` = offload everything |
| `threads` | `--threads` | CPU threads |
| `parallel` | `--parallel` | concurrent request slots |
| `batch_size` | `--batch-size` | prefill batch |
| `ubatch_size` | `--ubatch-size` | prefill micro-batch |
| `cache_type_k` | `--cache-type-k` | KV cache dtype for K, e.g. `q8_0`, `f16` |
| `cache_type_v` | `--cache-type-v` | KV cache dtype for V |
| `spec_type` | `--spec-type` | speculative decoding mode, e.g. `draft-mtp`, `draft-simple` |
| `spec_draft_n_max` | `--spec-draft-n-max` | draft lookahead length |
| `spec_draft_ngl` | `--spec-draft-ngl` | standalone draft model only: layers of it to keep in VRAM |
| `reasoning_effort` | `--reasoning-effort` | `minimal\|low\|medium\|high\|xhigh\|max` (template-dependent) |
| `chat_template_kwargs` | `--chat-template-kwargs` | JSON object of extra jinja variables for templates that gate thinking on their own key, e.g. `'{"reasoning_strength":"low"}'` (Muse-Glimmer) |
| `image_min_tokens` | `--image-min-tokens` | vision: min tokens per image |
| `image_max_tokens` | `--image-max-tokens` | vision: max tokens per image |
| `n_cpu_moe` | `--n-cpu-moe` | MoE only: keep first N layers' experts in system RAM |
| `temperature` | `--temperature` | sampling default (llama.cpp default: `0.80`) |
| `top_p` | `--top-p` | sampling default (default: `0.95`) |
| `top_k` | `--top-k` | sampling default (default: `40`) |
| `min_p` | `--min-p` | sampling default (default: `0.05`) |
| `presence_penalty` | `--presence-penalty` | sampling default (default: `0.00`) |
| `repeat_penalty` | `--repeat-penalty` | sampling default (default: `1.00`) |
| `chat_template` | `--chat-template` | override the GGUF's built-in Jinja template |

### Boolean flags (present only if truthy)

| key | CLI flag |
|---|---|
| `cont_batching` | `--cont-batching` |
| `mlock` | `--mlock` |
| `no_mmap` | `--no-mmap` |
| `embedding` | `--embedding` (legacy; prefer `modality: embedding`) |
| `no_mmproj_offload` | `--no-mmproj-offload` — keeps the vision projector on CPU instead of VRAM; frees the mmproj's own footprint (and its compute buffers) for a larger `ctx_size`, at the cost of slower image encoding |

### Tri-state (`true`/`false`/`"on"`/`"off"`/`"auto"`)

| key | CLI flag |
|---|---|
| `flash_attn` | `--flash-attn` |
| `reasoning` | `--reasoning` |

---

## `embedding` (backend: `embedding-server`)

Same `LlamaServerBackend` param tables as `text` above (it subclasses it), plus
`--embedding` is always forced on — you don't need to set the `embedding` flag
yourself. Use `modality: embedding` rather than the legacy `embedding: true`
param on a `text` entry.

`ctx_size` is not the per-request cap here: embedding is non-causal, so each
input must fit in a single `ubatch_size` (llama.cpp requires `ctx_size >=
ubatch_size`, but `ctx_size` can be much larger without raising the actual
per-call limit). Leaving `batch_size`/`ubatch_size` at the llama.cpp default
(2048/512) silently truncates any input over 512 tokens even if `ctx_size` is
set to the model's full native context. Set `batch_size`/`ubatch_size`
explicitly to your largest expected chunk length, and keep `ctx_size` aligned
to the same figure (rather than the model's advertised max) so the config
isn't promising a capacity it can't actually serve per call.

## `rerank` (backend: `rerank-server`)

Same `LlamaServerBackend` param tables as `text` above (it subclasses it), plus
`--reranking` is always forced on. Mutually exclusive with `--embedding`
upstream — don't set `embedding: true` on a rerank entry.

---

## `asr` (backend: `parakeet-server`)

Source: `ParakeetServerBackend`. `/v1/audio/transcriptions`, multipart WAV
upload.

| key | CLI flag |
|---|---|
| `threads` | `--threads` |
| `cache_dir` | `--cache-dir` |

---

## `asr` (backend: `parakeet-rs-server`) — ASR + speaker diarization

Source: `ParakeetRsServerBackend`, wrapping `parakeet-rs-server` from
[rzafiamy/parakeet-rs](https://github.com/rzafiamy/parakeet-rs) (`server/`).
Build with `./build-parakeet-rs.sh`. Set `backend: parakeet-rs-server`
explicitly — `parakeet-server` stays the default for `modality: asr`.

Serves `/v1/audio/transcriptions` (`json`, `text`, `verbose_json`, `srt`,
`vtt`, `diarized_json`; `diarize=true`; `stream=true`) and
`/v1/audio/diarize` (`json` or `rttm`). Uploads are forwarded untouched —
no WAV transcode, no `ZALLAMA_ASR_SILENCE_CAP` clamp — because the server
decodes every format itself and must see the original timing.

`file` is the ONNX model **directory** (`encoder-model.onnx` [+ `.data`],
`decoder_joint-model.onnx`, `vocab.txt`). The diarization model is an
artifact:

| artifact | CLI flag |
|---|---|
| `diarization` | `--diarization-model` (Nemotron-3 Diarization `.onnx`) |

| key | CLI flag | default |
|---|---|---|
| `threads` | `--threads` | min(8, cores) |
| `device` | `--device` (`auto`, `cpu`, `cuda`) | `auto` |
| `device_id` | `--device-id` | 0 |
| `gpu_mem_limit_mb` | `--gpu-mem-limit-mb` (per ONNX session) | 0 = none |
| `diarization_device` | `--diarization-device` (`auto` follows `device`) | `auto` |
| `diar_onset` / `diar_offset` | `--diar-onset` / `--diar-offset` | 0.5 / 0.5 |
| `diar_min_duration_on` / `diar_min_duration_off` | `--diar-min-duration-on` / `-off` (s) | 0 / 0 |
| `max_chunk_secs` | `--max-chunk-secs` | 120 |
| `split_silence_secs` | `--split-silence-secs` (0 = off) | 1.0 |
| `max_audio_secs` | `--max-audio-secs` (0 = unlimited) | 0 |
| `max_upload_mb` | `--max-upload-mb` | 512 |
| `max_queue` | `--max-queue` | 16 |

`mem_gb`: fp16 on CUDA peaks at 2.1 GB with `diarization_device: cpu` and
2.6 GB with diarization on the GPU (the fp32 export needs about twice that).
CPU entries use no VRAM, so leave `mem_gb` unset there.

---

## `asr` (backend: `audiocpp-server`) — Voxtral Mini 4B Realtime

Source: `AudioCppServerBackend`, wrapping [mirek190/audio.cpp](https://github.com/mirek190/audio.cpp)'s
server. Build with `./build-ggml-audio.cpp.sh`. Set `backend: audiocpp-server`
explicitly on the registry entry — `parakeet-server` stays the default for
`modality: asr`.

`/v1/audio/transcriptions` (same multipart contract as parakeet-server, so
zallama's existing route proxies it unchanged). audio.cpp also exposes a true
streaming `POST /v1/audio/transcriptions/live` endpoint upstream, which
zallama does not proxy today.

audio.cpp's server is config-file driven rather than pure-CLI: the backend
writes a small single-model JSON config as a sibling of the model directory
(`<model_dir_parent>/.audiocpp-<name>.json`) on every spawn, then drives
host/port/backend/etc. from real CLI overrides.

| key | CLI flag |
|---|---|
| `threads` | `--threads` |
| `device` | `--device` |
| `busy_timeout_ms` | `--busy-timeout-ms` |

`file` must point at a **directory** containing the `.gguf` (config/tokenizer
metadata is embedded in the GGUF itself — the HF repo
`audio-cpp/audio.cpp-gguf`'s `Voxtral-Mini-4B-Realtime-2602-GGUF/` prefix
ships no separate sidecar files).

---

## `tts` (backend: `kokoro-server`)

Source: `KokoroServerBackend`. `/v1/audio/speech`, JSON in / WAV out.

kokoro-server's CLI takes only `--model`/`--host`/`--port` — it has **no
launch-time synthesis flags**. `voice` and `speed` are request-body-only
fields; the registry's `params.voice`/`params.speed` are applied by the
`/v1/audio/speech` route as defaults when the client's request omits them.

| key | applied where | notes |
|---|---|---|
| `voice` | request body default | precedence: request `voice` > text-language auto-detect > `params.voice` > kokoro's own default |
| `speed` | request body default | applied only if the request omits `speed` |

`file` for a kokoro entry must point at the model's **resource directory**
(two ONNX models + a voice pack), not a single weights file.

---

## `tts` (backend: `voxtral-tts-server`) — Voxtral-4B-TTS-2603

Source: `VoxtralTtsServerBackend`, a thin server of ours
(`patches/voxtral-tts-server.cpp`) on top of
[mudler/voxtral-tts.c](https://github.com/mudler/voxtral-tts.c)'s real C/CUDA
inference engine. Build with `./build-voxtral-tts.sh`. Set
`backend: voxtral-tts-server` explicitly — `kokoro-server` stays the default
for `modality: tts`.

Same CLI shape and endpoint contract as `kokoro-server`: `--model`/`--host`/
`--port`, `POST /v1/audio/speech` (JSON in: `input`, optional `voice` —
`speed` isn't supported by the engine and is ignored if sent), `GET /health`.
No launch-time params beyond the path.

`file` must point at the model directory containing `consolidated.safetensors`
+ `tekken.json`.

> **License:** Voxtral-4B-TTS-2603 weights are Mistral AI's, under
> [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) —
> **non-commercial use only**. The server code here (and the upstream engine
> it wraps) is MIT.
>
> Unlike some third-party wrappers of the same engine, this server has **no
> silent fallback**: if the model fails to load, the process exits non-zero
> instead of serving silent audio.

---

## `image` (backend: `sd-server`)

Source: `SdServerBackend`. `/v1/images/generations`, JSON in / JSON
(`b64_json`/`url`) out. Only the flags listed here are forwarded — anything
else in `params` is dropped (`sd-server` aborts on an unrecognized argument).

### Value params

| key | CLI flag | notes |
|---|---|---|
| `model_flag` | *(selects primary flag)* | `"-m"` (checkpoint) vs `"--diffusion-model"` (standalone GGUF diffusion weights); auto-picked from file extension if unset |
| `threads` | `--threads` | |
| `vae` | `--vae` | overridden if `artifacts.vae` is set |
| `taesd` | `--taesd` | |
| `control_net` | `--control-net` | |
| `clip_l` | `--clip_l` | |
| `clip_g` | `--clip_g` | |
| `t5xxl` | `--t5xxl` | |
| `llm` | `--llm` | text encoder for newer archs (Qwen2.5-VL, Mistral-Small-3.2) |
| `vae_format` | `--vae-format` | |
| `steps` | `--steps` | also a request-body default (see below) |
| `cfg_scale` | `--cfg-scale` | also a request-body default |
| `sampler` | `--sampling-method` | also a request-body default |
| `scheduler` | `--scheduler` | |
| `width` | `--width` | |
| `height` | `--height` | |
| `seed` | `--seed` | |
| `vae_tile_size` | `--vae-tile-size` | only meaningful with `vae_tiling: true` |
| `vae_tile_overlap` | `--vae-tile-overlap` | only meaningful with `vae_tiling: true` |
| `vae_relative_tile_size` | `--vae-relative-tile-size` | overrides `vae_tile_size`; fractions of image size when < 1, tiles per dim when >= 1 |
| `backend` | `--backend` | per-component device placement, e.g. `"vae=cuda0,diffusion=cpu"` |
| `params_backend` | `--params-backend` | |
| `cache_mode` | `--cache-mode` | step caching: `easycache`, `ucache`, `dbcache`/`taylorseer`/`cache-dit`, `spectrum`. Nothing to gain below ~20 steps |
| `cache_option` | `--cache-option` | named cache params, e.g. `"threshold=0.25"` |
| `max_vram` | `--max-vram` | GiB budget for graph-cut segmented execution; negative auto-detects free VRAM sparing that much |
| `split_mode` | `--split-mode` | `layer` or `row` when a module spans several devices |
| `rpc_servers` | `--rpc-servers` | |
| `type` | `--type` | cast every weight on load, e.g. `q8_0` |
| `tensor_type_rules` | `--tensor-type-rules` | per-pattern load-time typing, e.g. `"^vae\.=f16,model\.=q8_0"` |
| `model_args` | `--model-args` | arch-specific key=value list |
| `guidance` | `--guidance` | distilled guidance scale (models with a guidance input) |
| `img_cfg_scale` | `--img-cfg-scale` | inpaint / image-edit models |
| `flow_shift` | `--flow-shift` | flow models (SD3.x, WAN) |
| `eta` | `--eta` | noise multiplier |
| `sigmas` | `--sigmas` | explicit comma-separated sigma schedule |
| `clip_skip` | `--clip-skip` | |
| `batch_count` | `--batch-count` | |
| `timestep_shift` | `--timestep-shift` | NitroFusion models |
| `extra_sample_args` | `--extra-sample-args` | sampler/scheduler/guidance key=value list |
| `extra_tiling_args` | `--extra-tiling-args` | VAE tiling key=value list |
| `slg_scale` | `--slg-scale` | skip layer guidance, DiT only; `0` disables |
| `skip_layers` | `--skip-layers` | layers SLG skips (default `[7,8,9]`) |
| `skip_layer_start` / `skip_layer_end` | `--skip-layer-start` / `--skip-layer-end` | SLG window |
| `strength` | `--strength` | noising/unnoising strength |
| `rng` / `sampler_rng` | `--rng` / `--sampler-rng` | `std_default`, `cuda`, `cpu` |
| `prediction` | `--prediction` | prediction type override |
| `lora_model_dir` | `--lora-model-dir` | |
| `lora_apply_mode` | `--lora-apply-mode` | `auto`, `immediately`, `at_runtime` |
| `embd_dir` | `--embd-dir` | textual-inversion embeddings |
| `hires_upscaler` | `--hires-upscaler` | with `hires: true` |
| `hires_upscalers_dir` | `--hires-upscalers-dir` | |
| `hires_scale` | `--hires-scale` | used when `hires_width`/`hires_height` are unset |
| `hires_width` / `hires_height` | `--hires-width` / `--hires-height` | |
| `hires_steps` | `--hires-steps` | second-pass steps; `0` reuses `steps` |
| `hires_sigmas` | `--hires-sigmas` | |
| `hires_denoising_strength` | `--hires-denoising-strength` | |
| `hires_upscale_tile_size` | `--hires-upscale-tile-size` | |
| `upscale_model` | `--upscale-model` | ESRGAN weights |
| `upscale_repeats` | `--upscale-repeats` | |
| `upscale_tile_size` | `--upscale-tile-size` | |

Artifacts take priority over the matching `params` key of the same name and are
passed as their own flag: `artifacts.vae`, `.taesd`, `.audio_vae`,
`.control_net`, `.clip_l`, `.clip_g`, `.t5xxl`, `.llm`, `.llm_vision`,
`.clip_vision`, `.ip_adapter`, `.photo_maker`, `.pulid_weights`,
`.motion_module`, `.upscale_model`, `.high_noise_diffusion_model`,
`.uncond_diffusion_model`, `.embeddings_connectors`.

> **The text encoder is usually the VRAM problem, not the diffusion model.**
> A FLUX stack with `t5xxl_fp16.safetensors` spends 9.8 GB on the encoder
> against 6.8 GB on Q4_0 diffusion weights. Registering a quantized encoder
> instead (`t5-v1_1-xxl-encoder-Q8_0.gguf`, 5.1 GB) took a measured stack from
> 17.0 to 11.8 GiB with no visible quality change — see
> [docs/sd-tuning.md](docs/sd-tuning.md).

### Boolean flags (present only if truthy)

| key | CLI flag | notes |
|---|---|---|
| `vae_tiling` | `--vae-tiling` | avoids an OOM on the VAE decode buffer at large resolutions — but costs ~20% of the clock, so measure before enabling |
| `fa` | `--fa` | flash attention everywhere. Measured to add nothing over `diffusion_fa` alone |
| `diffusion_fa` | `--diffusion-fa` | flash attention in the diffusion model. Worth **1.73x** on FLUX at 1024x1024 |
| `diffusion_conv_direct` | `--diffusion-conv-direct` | measured as noise on FLUX |
| `vae_conv_direct` | `--vae-conv-direct` | saves ~0.5 GiB and costs **2.7x** the wall time — see [docs/sd-tuning.md](docs/sd-tuning.md) |
| `offload_to_cpu` | `--offload-to-cpu` | keep weights in RAM, stream into VRAM per graph. Prefer `max_vram` + `stream_layers` |
| `stream_layers` | `--stream-layers` | residency + prefetch on top of `max_vram`; no effect without it |
| `auto_fit` | `--auto-fit` | derive placement from model size and per-device budget; overrides `backend`/`params_backend` |
| `eager_load` | `--eager-load` | load all params at model-load time instead of lazily on first use, moving the cost into the startup health check |
| `mmap` | `--mmap` | memory-map the weights |
| `hires` | `--hires` | highres fix: sample small, upscale, re-denoise |
| `temporal_tiling` | `--temporal-tiling` | LTX video VAE decode |
| `circular` / `circularx` / `circulary` | `--circular` / `--circularx` / `--circulary` | circular padding for tileable output |
| `force_sdxl_vae_conv_scale` | `--force-sdxl-vae-conv-scale` | |
| `disable_image_metadata` | `--disable-image-metadata` | |
| `increase_ref_index` | `--increase-ref-index` | |
| `disable_auto_resize_ref_image` | `--disable-auto-resize-ref-image` | |

### Request-body-only defaults

Applied by the `/v1/images/generations` route when the client's request omits
them — these are **not** CLI flags:

| key | notes |
|---|---|
| `steps` | also settable as a launch-time CLI default, see table above |
| `cfg_scale` | also settable as a launch-time CLI default |
| `sampler` | also settable as a launch-time CLI default |
| `negative_prompt` | request-body only, no CLI equivalent |

---

## Daemon-wide config (`config.yaml`, not per-model)

Not part of the registry, but the other half of the merge chain — see
`llama_server.default_params` in `config.yaml`/`config.example.yaml`: any key
from the `text`/`embedding`/`rerank` tables above set there applies to every
`llama-server`-family model unless overridden by that model's own `params`.
`llama_server.mem_budget_gb`, `.max_loaded_models`, `.idle_timeout`,
`.port_start`, `.startup_timeout`, `.evict_drain_timeout` control process
lifecycle, not per-model launch flags, and have no `params` equivalent.

`zallama.port` is the inference listener (`/v1/*`); `zallama.admin_port`
(default `port + 1`) carries the management API (`/api/*`) and Prometheus
`/metrics`, optionally on its own interface via `zallama.admin_host`. See
`docs/monitoring.md`.
