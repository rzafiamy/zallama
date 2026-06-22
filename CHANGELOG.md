# Changelog

All notable changes to **Zallama** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Pluggable backend abstraction** (`server/backends.py`): a `Backend` protocol isolates
  engine-specific logic (binary, argument building, health path) from the generic process
  lifecycle, so new modalities can be added as new backends rather than cross-cutting changes.
- **Modality-aware registry**: model entries may declare `modality` (`text`, and — by design —
  `asr` / `tts` / `image`), `backend`, and `artifacts`. All fields are optional and default to
  the classic single-GGUF `llama-server` text model, so existing registries keep working.
- **Vision (multimodal) support**: attach an `mmproj` projector via `artifacts`; Zallama passes
  `--mmproj` to `llama-server` automatically and image input flows through `/v1/chat/completions`.
- **Modality guard**: requests to an endpoint a model cannot serve return a clear `400` instead
  of a confusing upstream failure. The audio/image endpoints are pre-mapped, awaiting backends.
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
- **Ollama-like CLI**: `serve`, `list`/`ls`, `add`, `set`, `pull`, `search`, `remove`/`rm`,
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

[Unreleased]: https://github.com/rzafiamy/zallama/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rzafiamy/zallama/releases/tag/v1.0.0
