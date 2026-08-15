# Model Tuning Log

*One row per measured configuration — what changed, what it cost, what it bought.
A running record so a tuning session never has to be redone from scratch.*

Every row here was measured, not estimated — see
[Measuring what a model actually costs](vram-planning.md#measuring-what-a-model-actually-costs)
for how. Deep dives on a single knob get their own doc (e.g.
[Doubling Decode Speed with MTP](mtp-speculative-decoding.md)); this page is the
flat index across all of them.

- [Log](#log)
- [Adding a row](#adding-a-row)

---

## Log

| Model | Arch | GPU | Config | Decode tok/s | VRAM | Date | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-Q4_K_M | qwen35 (hybrid) | RTX 4090 24 GiB | `ctx_size 8192`, f16 KV, no MTP (baseline) | 47.9 | 17.3 GiB | 2026-08-14 | Truncating conversations at 8192. |
| Qwen3.8-27B-Q4_K_M | qwen35 (hybrid) | RTX 4090 24 GiB | `ctx_size 65536`, `cache_type_k/v q8_0`, `spec_type draft-mtp`, `spec_draft_n_max 3` | **106.5** greedy / ~112.5 @ temp 1.0 top_p 0.95 top_k 20 | 20.3 GiB | 2026-08-14 | 2.2x baseline; 71% draft acceptance at sampled temp. Full writeup: [mtp-speculative-decoding.md](mtp-speculative-decoding.md). |
| Qwen3.8-27B-Q4_K_M | qwen35 (hybrid) | RTX 4090 24 GiB | `spec_draft_n_max` sweep: 2 / 3 / 4 / 5 / 6 / 8, else as above | 95.8 / **106.5** / 102.0 / 97.5 / 92.2 / 91.9 | 20.1–20.7 GiB | 2026-08-14 | Fixed 3-prompt set, `--temp 0`. 3 is llama.cpp's default and wins here. |
| Qwen3.8-27B-Q4_K_M | qwen35 (hybrid) | RTX 4090 24 GiB | `cache_type_k/v`: f16 vs q8_0, else as winning config | 109.3 vs 106.1 | +874 MiB for f16 | 2026-08-14 | 3% decode gain not worth halving the context that fits. Kept q8_0. |
| Qwen3.8-27B-Q4_K_M | qwen35 (hybrid) | RTX 4090 24 GiB | `ubatch_size`: 512 / 1024 / 2048 (prefill) | 2493 / 2491 / 2455 tok/s prefill | up to +0.8 GiB | 2026-08-14 | Prefill already compute-bound; knob is inert here. |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 | nemotron_h_moe (hybrid MoE) | RTX 4090 24 GiB | `ctx_size 444416`, `cache_type_k/v q8_0`, `spec_type draft-mtp` (nextn head baked into the main GGUF) | **332.4** greedy | 21.3 GiB | 2026-08-15 | 80–92% draft acceptance at temp 0. Only 7 of 53 blocks cache anything (rest is Mamba/SSM) and KV-heads = 2, so the KV cache costs ~4 KiB/token — `calibrate` (after the fix below) recommended the max the trained ctx allows. A same-repo `mtp-*.gguf` sibling turned out to be a redundant copy of the already-embedded nextn block, not a separate draft model — checked its tensor list before wiring `--model-draft`. |
| Muse-Glimmer-30B-UD-Q4_K_XL | muse-glimmer (dense, SWA) | RTX 4090 24 GiB | `ctx_size 33792`, `cache_type_k/v q8_0`, vision mmproj (+3.6 GiB) | 52.8 | 18.8 GiB | 2026-08-15 | Dense model, no MTP head to exploit. `calibrate` doesn't parse this arch's `attention.sliding_window_pattern` (published as a period int, not Gemma3's per-layer bool array) so it prices every layer as full-context — the 33792 recommendation is conservative; real max ctx is likely well above it. Not fixed yet, see below. |

**2026-08-15 — two calibrate/registry gaps found registering the above:**
- `_gguf_arch_dims` (`zallama` CLI) assumed `attention.head_count_kv` is a scalar.
  Nemotron-H publishes it as a **per-layer array** (0 = SSM layer, nonzero =
  attention layer) instead of the scalar + `full_attention_interval` pair
  Qwen3.x uses for the same idea. The old code raised inside the dims
  calculation, which the caller swallows into the crude size-only fallback
  (ctx 4096) — no crash, just silently wrong. Fixed by reading the list
  directly: `n_kv_layer` = count of nonzero entries (the trailing MTP block is
  itself a nonzero entry, so it's included for free, unlike the interval path
  which has to add it back explicitly).
- `backends.py` had no way to pass `--model-draft` (a genuinely separate draft
  GGUF, as opposed to `draft-mtp`'s head baked into the main file). Added a
  `draft` artifact key → `--model-draft`, plus `spec_draft_ngl` →
  `--spec-draft-ngl`. Unused by either model above (both either have no MTP
  head or already carry it embedded) but needed the next time a model ships
  with a genuinely external draft checkpoint. Requires a `zallama` service restart to
  take effect (`backends.py` loads once into the long-running daemon, unlike
  the CLI script which re-execs fresh every call).
- Still open: teach `calibrate` the scalar-period form of
  `attention.sliding_window_pattern` (Muse-Glimmer's case above) the way it
  already handles Gemma3's per-layer boolean array.

## Adding a row

1. **Fix the comparison.** Use the same prompt set and `--temp 0` for anything
   you're ranking against another value — `zallama bench`'s default (varying,
   sampled output) has enough spread to flip a ranking; see the warning in
   [mtp-speculative-decoding.md](mtp-speculative-decoding.md#tuning-the-lookahead).
2. **Record the config that produced the number**, not just the winner — a
   losing value is what stops the next session from re-testing it.
3. **VRAM from `zallama ps`**, not a calculated estimate, and feed the result
   back into `mem_gb` via `zallama set <model> mem_gb=...` so eviction
   schedules on the real number.
4. One row per *change*, not per run — average a few runs first if
   `zallama bench --runs` shows meaningful spread, and put the ± in the cell.
5. Link out to a deep-dive doc when a topic grows past a table row (MTP did).
