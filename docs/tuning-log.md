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
