# Changelog

All notable changes to **Zallama** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.0] - 2026-08-15

### Added
- `draft` artifact and `spec_draft_ngl` param for `text`/`embedding`/`rerank` entries, mapping to
  `llama-server`'s `--model-draft` / `--spec-draft-ngl`. This is standalone speculative decoding —
  a genuinely separate, smaller draft GGUF (`spec_type: draft-simple` and friends) — as opposed to
  `draft-mtp`, whose draft head is already baked into the main GGUF and needs neither flag. Before
  this there was no way to register a model's external draft checkpoint at all.

### Fixed
- `zallama calibrate` crashed its own GGUF-dims calculation (silently, into the crude size-only
  fallback) on architectures that publish `attention.head_count_kv` as a **per-layer array**
  rather than a scalar. Nemotron-H marks each of its blocks attention (nonzero) or SSM/Mamba (0)
  this way, instead of Qwen3.x's scalar `head_count_kv` + `full_attention_interval` pair for the
  same idea. `_gguf_arch_dims` now reads the array directly — an exact per-layer count of
  KV-caching blocks, rather than an assumed even interval — recovering full-size recommendations
  (444k tokens of context fit a Nemotron-H MoE that the old fallback would have capped at 4096).

### Known gaps
- `calibrate` still doesn't parse the scalar-period form of `attention.sliding_window_pattern`
  (an int meaning "1 in every N layers is global", used by e.g. Muse-Glimmer) — only Gemma3's
  per-layer boolean array. It falls back to pricing every layer as full-context, which is safe
  (never over-recommends) but conservative. See [Model Tuning Log](docs/tuning-log.md).

## [1.8.0] - 2026-08-15

### Added
- `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty` and `repeat_penalty` model params,
  each mapped to the matching `llama-server` CLI flag. Previously these were silently dropped —
  `_PARAM_MAP` had no entry for any of them — so a registered value never reached the model and
  every model ran on llama.cpp's own hardcoded sampling defaults regardless of what its model card
  recommended. Because `llama-server` only falls back to its CLI default when a chat request
  omits the field, this gets registry-level sampling defaults with the priority chain
  `request body > registry params > config default_params > llama.cpp's built-in default` for
  free, with no proxy-side merge logic needed.
- [CONFIG.md](CONFIG.md): a full reference of every `params` key each backend/modality accepts
  (`text`/`embedding`/`rerank` on `llama-server`, `asr` on `parakeet-server`, `tts` on
  `kokoro-server`, `image` on `sd-server`), which ones are CLI launch flags vs. request-body-only
  defaults, and the merge/priority chain between `config.yaml`, `registry.yaml`, and the request.

## [1.7.0] - 2026-08-15

### Fixed
- **`zallama calibrate` under-sized the context by 6x on hybrid architectures.** It assumed every
  block holds a KV cache and derived `head_dim` from `n_embd / n_head`. Neither holds on models
  that interleave linear/SSM layers with full attention: Qwen3.8-27B publishes
  `full_attention_interval: 4` (16 of its 64 blocks cache anything) and decoupled
  `attention.key_length` / `value_length` of 256 against an `n_embd / n_head` of 213. The
  recommendation for that model was `ctx_size=10240` where 65536 fits and was measured at 20.3 GiB.
  Calibration now counts only the caching layers (plus the MTP `nextn` block, which does get a
  cache once `--spec-type draft-mtp` loads it), reads the real K and V widths, adds the artifacts
  (a vision mmproj is ~1 GB of VRAM the budget ignored), and sizes against the model's configured
  `cache_type_k`/`cache_type_v` instead of always assuming f16. The report shows the per-token cost
  it derived, and the whole calculation is now in GiB — it mixed GiB from `nvidia-smi` with
  decimal-GB file sizes, shrinking every budget by a further 7.4%.
- **`zallama set <model> mem_gb=...` wrote the value where nothing reads it.** `mem_gb` landed
  inside `params`; the memory budget reads it from the entry level, so the declared cost was
  silently ignored and the model kept being scheduled on the `file_size * 1.2` estimate — while
  `zallama calibrate` ended by telling you to run exactly that command. `mem_gb`, `pinned` and
  `description` are now recognised as entry-level fields (like `modality` and `backend`), are
  reported when they change, and a stale copy left in `params` by an older version is cleaned up
  on the next `set`.
- **`zallama list` / `show` / `ps` served stale data after a manual registry edit.**
  `ModelRegistry.list_models()` never reloaded, unlike `get()`, so a hand-edited `registry.yaml`
  stayed invisible until some inference request happened to go through `get()`.

### Added
- `reasoning_effort` model param (`--reasoning-effort`). Thinking models whose chat template
  defaults to a high effort — Qwen3.8's starts at `xhigh` — spend most of their wall-clock inside
  `<think>`, where dropping the effort saves more real time than any decoding knob.
- `image_min_tokens` / `image_max_tokens` model params. llama.cpp warns at load that Qwen-VL
  models need at least 1024 image tokens for grounding tasks to stay accurate; there was no way to
  set it from the registry.
- [Doubling Decode Speed with a Baked-In MTP Head](docs/mtp-speculative-decoding.md): how to spot
  an unused `nextn` head in a GGUF, what `spec_type: draft-mtp` costs and returns, why
  `spec_draft_n_max` peaks at 3, and why `zallama bench`'s default workload is too noisy to
  settle that question on its own.

## [1.6.0] - 2026-08-09

### Fixed
- **TTS was reading French with English sounds.** kokoro takes no language argument — it
  phonemizes according to the *voice prefix* (`ff_siwis` → `fr-FR`, `af_heart` → `en-US`). The
  registry default `voice: af_heart` was applied to every request that omitted `voice`, so French
  input was phonemized as English: the same French sentence yields 6.29 s of audio through
  `af_heart` against 5.21 s through `ff_siwis`, the difference being English phonemes spelled out
  over French words. When a request names no voice, Zallama now infers the language from the text
  and picks that language's voice (`server/tts_lang.py`). Precedence is **request voice → detected
  language → registry `voice` param → kokoro's own default**, so an explicit voice is never
  overridden and the registry default still covers text that can't be placed.
- **TTS was ~2.5x slower than it had to be, silently.** kokoro loads libespeak-ng through
  `dlopen` for grapheme-to-phoneme and falls back — with no error — to a bundled ByT5 model that
  is autoregressive with no KV cache, ~50 ms per generated phoneme. espeak-ng was never listed as
  a dependency, so the slow path was the only path. `build-ggml-kokoro.cpp.sh` now installs it,
  and `install.sh` warns when `kokoro-server` is present without it. Note that the packages
  `libpcaudio0` / `libsonic0` are needed too: they are libespeak-ng's own shared dependencies and
  the `dlopen` fails silently without them.

### Changed
- **kokoro.cpp pinned to v0.3.0** (was v0.1.0). Measured on an i9-14900K, 1248 characters of
  English:

  | Build | audio produced | wall | × realtime | CPU |
  |---|---:|---:|---:|---:|
  | v0.1.0 | 22.9 s *(truncated)* | 65.8 s | 0.35× | 1619 s |
  | v0.3.0 + espeak-ng | 85.9 s | 12.5 s | **6.9×** | **58 s** |

  v0.1.0 also truncated long input, emitting 22.9 s of audio where v0.3.0 emits 85.9 s from the
  same text, so the throughput gain is larger than the wall-clock column alone suggests. v0.3.0
  additionally caps ONNX Runtime's thread pools (`--threads`, default `min(cores, 8)`): the old
  defaults sized the intra-op pool to the logical core count and spun between operators, burning
  331 s of CPU where 58 s does the same work — which matters when llama-server shares the box.

### Documented
- **A Text-to-Speech section in the README**, which had none despite sections for ASR and image
  generation: endpoint usage, the voice-selection precedence table, and the espeak-ng requirement.
  The installation section now also records that kokoro needs CMake 3.29+ and picks espeak-ng up
  at runtime.

## [1.5.1] - 2026-08-08

### Fixed
- **Image generation no longer dies at the final VAE decode**: `SdServerBackend` mapped only
  value-taking flags, so no boolean `sd-server` option was reachable from the registry — including
  `--vae-tiling`. The VAE decode buffer grows with the square of the image size and is allocated
  *after* the whole stack is resident in VRAM: a FLUX 1024×1024 decode asks for 6.66 GB on top of
  15.94 GB of weights, overflowing a 24 GB card and failing the request after every sampling step
  had already succeeded (`cudaMalloc failed: out of memory` → `decode_first_stage failed`). The
  backend now has a `_FLAG_MAP` alongside its `_PARAM_MAP`, and `vae_tiling` decodes the latent in
  patches for a few hundred MB instead.

### Added
- **Memory switches for `sd-server`**: `vae_tiling`, `fa`, `diffusion_fa`, `diffusion_conv_direct`,
  `vae_conv_direct` and `offload_to_cpu` (weights held in RAM, streamed into VRAM per graph), plus
  the value params `vae_tile_size`, `vae_tile_overlap`, and `backend` / `params_backend` for
  per-component device placement (`te=cpu`, `vae=cuda0,diffusion=cpu`) — the supported replacement
  for the deprecated `--clip-on-cpu` / `--vae-on-cpu`. Documented in the README's image section,
  with `vae_tiling` enabled on the example image model.

## [1.5.0] - 2026-08-08

### Added
- **`n_cpu_moe` param — fit a MoE model that would otherwise not load**: maps to llama-server's
  `--n-cpu-moe N`, keeping the expert weights of the first N layers in system RAM while attention
  and the KV cache stay on the GPU. It is the finest-grained VRAM knob for Mixture-of-Experts
  models, and unlike the alternatives it costs neither context nor vision. Measured on
  Qwen3.6-35B-A3B (RTX 4090, ctx 131072, mmproj loaded): `N=0` → 23.44 GiB / 178 tok/s,
  `N=4` → 21.77 GiB / 138 tok/s, `N=8` → 19.96 GiB / 113 tok/s — roughly 0.4 GiB freed and 8% of
  generation speed lost per offloaded layer. No effect on dense models.
- **`scripts/measure_vram.py` — measure what a model really costs**: builds each model's command
  line through Zallama's own `backend.build_args()`, launches it outside the daemon on a free
  port, waits for `/health`, reads the per-PID VRAM from `nvidia-smi`, and shuts it down.
  `--write` records the result as `mem_gb` in `registry.yaml`, editing only those lines so
  comments and key order survive. Special-cases `sd-server` (which allocates nothing until it
  generates, so a real 512×512 generation is triggered) and skips CPU-only `kokoro-server`.
- **`docs/vram-planning.md` — capacity planning guide**: how the scheduler decides what stays
  resident, why the built-in cost estimate is unreliable, how to measure, and the five levers for
  making a model fit (KV quantization, `ctx_size`, `n_cpu_moe`, `mmproj`, `n_gpu_layers`) with
  measured trade-offs for each, a worked example and a checklist.

### Documented
- **`pinned` is now documented**: the flag shipped earlier but appeared in no reference. The
  README gains a section covering what it is for (small always-on ASR/TTS backends: ~780 ms of
  reload per call becomes ~80 ms) and its two non-obvious behaviours — pinned models still count
  against `max_loaded_models`, and when everything loaded is pinned the incoming model is admitted
  **over budget** rather than evicting a warm pinned instance. Also added to
  `registry.example.yaml` and the registry reference.
- **`mem_budget_gb` now carries its precondition**: the fallback estimate (`gguf_size × 1.2`)
  ignores the KV cache, the artifacts (`mmproj`, `t5xxl`, `clip_l`, VAE) and the compute buffers,
  and is routinely wrong by more than 100% *in both directions* — a 4B model at `ctx_size:
  262144` measures 12.9 GiB against an estimated 4.2. Enabling a budget on such values evicts
  models that would have fit and admits models that then OOM, so `config.example.yaml`, the README
  and the guide now say to measure first.

## [1.4.0] - 2026-08-08

### Added
- **`zallama bench` — performance benchmarking across parameter sweeps**: measures prefill
  tok/s, decode tok/s, time-to-first-token, load time and VRAM for text models, and can sweep
  any llama-server parameter (`--sweep ctx_size=4096,16384,65536`, `--sweep reasoning=on,off`,
  repeat for a grid). Each combination is applied to the registry and the model restarted so it
  actually takes effect; the original params are restored when the run ends, including on Ctrl-C.
  Prefill/decode rates come from llama.cpp's own `timings` block (carried in the final SSE chunk),
  TTFT is measured client-side. Also supports prompt-size and concurrency sweeps
  (`-p 512,8192 -c 1,4`), repeated runs reported as mean ± stdev, `--all` to compare every
  registered text model, and `--out` export to `.json` / `.csv` / `.md`.
- **`zallama bench --compare a.json b.json`**: puts several exports side by side to compare
  machines (GPU vs GPU) on identical workloads. JSON exports now carry the machine that produced
  them — GPU model, VRAM, driver, CPU, host, and the llama.cpp build stamped into every response
  — and `--label NAME` names a run. Only points with identical model, swept params, prompt size,
  max-tokens and concurrency are compared; unmatched points are reported, and differing llama.cpp
  builds raise a warning. `--metric` selects which tables to print.
- **New sweepable llama-server params**: `batch_size`, `ubatch_size`, `cache_type_k` and
  `cache_type_v` are now recognized in a model's `params` (mapped to `--batch-size`,
  `--ubatch-size`, `--cache-type-k`, `--cache-type-v`).

### Fixed
- **VRAM is now labelled GiB, not GB, in `zallama bench`**: the value has always been
  `nvidia-smi` MiB / 1024, which is how a 24 GiB card gets mistaken for a 24 GB one.
- **`zallama set` no longer un-pins a pinned model**: `POST /api/models/add` accepts and
  preserves `pinned`, and `GET /api/models` reports it, so editing params on a pinned model
  keeps it pinned.
- **Long silences no longer truncate ASR transcriptions**: parakeet's TDT decoder skips past
  the speech following a long pause — 1.2 s transcribes in full, 1.6 s loses words, and 2.0 s
  returns an empty string. Audio sent to `/v1/audio/transcriptions` now passes through an
  ffmpeg `silenceremove` filter that clamps every pause to `ZALLAMA_ASR_SILENCE_CAP` seconds
  (default `0.8`; set `0` to disable). WAV input goes through the filter too, and is only
  passed through untouched when ffmpeg is unavailable. The filter keeps `stop_silence` at the
  cap rather than deleting pauses outright, which preserves speech rhythm and punctuation.

## [1.3.0] - 2026-08-06

### Added
- **Image generation backend (`sd-server`)**: integrated [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) as the `sd-server` backend powering the OpenAI-compatible `POST /v1/images/generations` endpoint and `zallama generate` CLI command. Added `./build-ggml-stable-diffusion.cpp.sh` build script.
- **FLUX, Z-Image, and Qwen-Image model support in Download Manager**: expanded `zallama pull` and the download manager with model definitions, artifact handling (diffusion models, CLIP L / T5xxl text encoders, VAEs), and optimized defaults for FLUX models (`flux:klein`, `flux:klein-9b`), Z-Image (`z-image:turbo`), and Qwen-Image (`qwen-image:20b`).
- **Deterministic document IDs in zvec**: updated document upsert in the `zvec` vector store to use deterministic SHA-256 hashes instead of random UUIDs.
- **Enhanced model addition validation**: `cmd_add` now validates model file path arguments and provides descriptive error messages.

### Fixed
- **Image generation execution path**: resolved model parameter loading and process execution issues for `sd-server`.

## [1.2.0] - 2026-08-03

### Added
- **RAG support**: retrieval-augmented generation built into Zallama.
  - **Reranking**: `RerankServerBackend` runs `llama-server` in `--reranking` mode (cross-encoder
    relevance scoring) behind a new `POST /v1/rerank` endpoint (Cohere/Jina-style response).
    Register reranker models with `modality: rerank, backend: rerank-server`.
  - **zvec vector store**: an in-process vector store backed by the [`zvec`](https://zvec.org)
    library (Alibaba, Apache-2.0) — an embedded HNSW vector DB, no external server. New
    `POST/GET/DELETE /v1/zvec/collections` and per-collection `upsert` / `query` / `delete`
    endpoints. Each collection is a directory under `rag.zvec_dir` (tracked by a manifest). zvec
    embeds and searches by calling Zallama's own `/v1/embeddings`, and a query can optionally
    rerank its candidates with a reranker model.
  - **`rag` config block + env overrides**: `rag.embedding_model`, `rag.rerank_model`,
    `rag.zvec_dir`, `rag.default_top_k` (env: `ZALLAMA_EMBEDDING_MODEL`, `ZALLAMA_RERANK_MODEL`,
    `ZALLAMA_ZVEC_DIR`, `ZALLAMA_RAG_TOP_K`).
  - **`zallama zvec` CLI**: `collections`, `create`, `drop`, `upsert <file>`, and
    `query <text> [--top-k] [--rerank]` subcommands.

### Changed
- **Embeddings are now a first-class modality**: `modality: embedding` on the new
  `EmbeddingServerBackend` (`llama-server --embedding`) serves `/v1/embeddings`, symmetric with
  `rerank`. The modality guard now protects `/v1/embeddings` (a chat model can't serve it, and an
  embedding model can't serve `/v1/chat/completions`). Legacy embedding models registered as
  `text` with `params: {embedding: true}` keep working — they are treated as `embedding` at
  runtime. The `nomic-embed:v1.5` shorthand now writes `modality: embedding`.
- **Configurable modality**: `zallama set <model> modality=<text|embedding|rerank|asr|tts|image>`
  now reclassifies an existing model, auto-selecting the matching backend (and `backend=`
  overrides the engine). `/api/models/add` validates modality and backend against the registered
  set and fills in the backend from the modality. The modality→backend mapping is centralized in
  `server/backends.py` (`MODALITY_BACKEND` / `default_backend_for`) and reused by the downloader.
- **README rewrite**: restructured around an "own your AI" narrative with a guided first-run
  walkthrough ("Your First 5 Minutes"), a features table, and a two-layer explanation of global
  vs. per-model configuration.

## [1.1.0] - 2026-06-23

### Added
- **Text-to-speech (TTS)**: `KokoroServerBackend` runs [kokoro.cpp](https://github.com/mudler/kokoro.cpp)
  models at the OpenAI-compatible `POST /v1/audio/speech` endpoint. Build the binary with
  `build-ggml-kokoro.cpp.sh` (copies `libggml*.so` and sets `RPATH=$ORIGIN`).
- **Model pinning & pre-warming**: models can be pinned so they are loaded eagerly and exempt
  from LRU/memory eviction, improving first-request latency for TTS and other hot models.
- **GGUF metadata reader & `calibrate` command**: reads model metadata directly from the GGUF
  header and derives recommended configuration (e.g. context/memory) for a model.
- **Speculative decoding**: `LlamaServerBackend` accepts draft-model settings for speculative
  decoding to accelerate generation.
- **Real VRAM monitoring**: `GET /api/ps` and `zallama ps` report actual GPU VRAM usage
  (queried from the GPU) rather than only declared/estimated cost.
- **Shell completions**: bash and zsh completion scripts for the `zallama` CLI.
- **`reload` command**: restarts running models to apply registry parameter changes without a
  full daemon restart.
- **Project website**: a Vite + Tailwind CSS site with interactive features.
- **Pluggable backend abstraction** (`server/backends.py`): a `Backend` protocol isolates
  engine-specific logic (binary, argument building, health path) from the generic process
  lifecycle, so new modalities can be added as new backends rather than cross-cutting changes.
- **Modality-aware registry**: model entries may declare `modality` (`text`, `asr`, and — by design —
  `tts` / `image`), `backend`, and `artifacts`. All fields are optional and default to
  the classic single-GGUF `llama-server` text model, so existing registries keep working.
- **Vision (multimodal) support**: attach an `mmproj` projector via `artifacts`; Zallama passes
  `--mmproj` to `llama-server` automatically and image input flows through `/v1/chat/completions`.
- **Speech-to-text (ASR)**: `ParakeetServerBackend` runs [parakeet.cpp](https://github.com/mudler/parakeet.cpp)
  models at the OpenAI-compatible `POST /v1/audio/transcriptions` endpoint. Non-WAV uploads are
  auto-transcoded to 16 kHz mono WAV via `ffmpeg`. `zallama pull` detects parakeet repos (and
  accepts `--type asr`) and registers them with the right modality/backend. Build the binary with
  `build-ggml-parakeet.cpp.sh` (copies `libggml*.so` and sets `RPATH=$ORIGIN`).
- **Modality guard**: requests to an endpoint a model cannot serve return a clear `400` instead
  of a confusing upstream failure. The remaining audio/image endpoints are pre-mapped, awaiting backends.
- **Memory-aware eviction**: optional `mem_budget_gb` evicts least-recently-used models to keep
  total declared/estimated cost within budget. Per-model cost comes from a declared `mem_gb`,
  else an estimate from GGUF file size, else `mem_init_gb`. Exposed via `GET /api/ps` and
  `zallama ps` (per-model memory + budget headroom).
- **Optional API-key auth**: when `zallama.api_key` is set, a Bearer token is required on the
  `/v1` and `/api` surfaces (health and Web UI stay public).
- **Configurable concurrency cap** `max_loaded_models` and configurable non-streaming
  `request_timeout`.
- `LICENSE` (MIT), this `CHANGELOG.md`, and an expanded README.

### Changed
- **Interactive chat markdown rendering**: the `run` chat now renders markdown with
  width-aware formatting in the terminal.
- **Web UI redesign**: updated to the Fluent 2 design language with an enhanced color palette
  and structured table layouts.
- **Default host is now `127.0.0.1`** (was `0.0.0.0`) — localhost-only by default; opt in to
  network exposure explicitly, ideally alongside `api_key`.
- **Per-model startup locking**: a slow model startup no longer blocks requests to other models
  (was a single global lock).
- **CORS** no longer combines wildcard origins with credentials (an invalid, ignored combination).
- Registry reads are mtime-cached (no longer re-parsing YAML on every inference request);
  registry writes are serialized to prevent concurrent `set`/`pull` clobbering.
- `resolve_binary` resolves per-backend executables (`./bin/<name>`, `~/.zallama/bin/<name>`, PATH).

### Fixed
- **PEP 668 install failure** (`externally-managed-environment`): `install.sh` now installs
  dependencies into a project-local `.venv`, and the `zallama` launcher transparently re-execs
  into it (override with `ZALLAMA_NO_VENV=1`). `sudo`-installed venvs are chowned back to the user.
- Port selection now bind-checks the OS so it won't hand out a port already held by a foreign
  process (previously surfaced as an opaque startup timeout).
- Half-started backend processes are killed on startup-timeout instead of lingering.
- CLI `set` preserves a model's `modality` / `backend` / `artifacts` / `mem_gb` instead of
  silently resetting them to defaults.

## [1.0.0] - 2026-06-22

Initial release.

### Added
- **Simple CLI**: `serve`, `list`/`ls`, `add`, `set`, `pull`, `search`, `remove`/`rm`,
  `run`, `ps`, `load`, `unload`/`stop`, `logs`, `health`, `version`.
- **OpenAI-compatible `/v1` API**: `chat/completions`, `completions`, `embeddings`,
  and `models`, with SSE streaming.
- **Dynamic process management**: on-demand `llama-server` spawn per model, automatic port
  assignment, `/health` readiness checking, and LRU eviction of idle models.
- **High-performance downloads**: `aria2c` (8 connections) with a concurrent HTTP-range
  fallback and a single-stream fallback; HuggingFace search and Unsloth shorthands.
- **Auto-detected GGUF quantization** (prefers Q4_K_M) when pulling a HuggingFace repo path
  without a specified filename.
- **Reasoning model support**: renders thinking/reasoning blocks in the interactive chat;
  `reasoning` is configurable per model.
- **Embedded Web UI** and a config-driven architecture (global defaults + per-model params).

[Unreleased]: https://github.com/rzafiamy/zallama/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/rzafiamy/zallama/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/rzafiamy/zallama/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/rzafiamy/zallama/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/rzafiamy/zallama/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/rzafiamy/zallama/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/rzafiamy/zallama/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/rzafiamy/zallama/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/rzafiamy/zallama/releases/tag/v1.0.0
