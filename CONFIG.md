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

If capacity is tight and no same-group victim is loaded, Zallama does **not**
fall back to evicting across groups — it logs a warning and admits the
incoming model over budget instead, the same fallback used when every loaded
model is pinned. See
[docs/vram-planning.md](docs/vram-planning.md) for a full worked example.

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
| `backend` | `--backend` | per-component device placement, e.g. `"vae=cuda0,diffusion=cpu"` |
| `params_backend` | `--params-backend` | |

Artifacts (`artifacts.vae`, `.taesd`, `.control_net`, `.clip_l`, `.clip_g`,
`.t5xxl`, `.llm`, `.llm_vision`, `.clip_vision`) take priority over the
matching `params` key of the same name and are passed as their own flag.

### Boolean flags (present only if truthy)

| key | CLI flag | notes |
|---|---|---|
| `vae_tiling` | `--vae-tiling` | avoids an OOM on the VAE decode buffer at large resolutions |
| `fa` | `--fa` | |
| `diffusion_fa` | `--diffusion-fa` | |
| `diffusion_conv_direct` | `--diffusion-conv-direct` | |
| `vae_conv_direct` | `--vae-conv-direct` | |
| `offload_to_cpu` | `--offload-to-cpu` | keep weights in RAM, stream into VRAM per graph |

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
`.port_start`, `.startup_timeout` control process lifecycle, not per-model
launch flags, and have no `params` equivalent.
