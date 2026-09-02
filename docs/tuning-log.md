# Model Tuning Log

*One row per measured configuration — what changed, what it cost, what it bought.
A running record so a tuning session never has to be redone from scratch.*

Every row here was measured, not estimated — see
[Measuring what a model actually costs](vram-planning.md#measuring-what-a-model-actually-costs)
for how. Deep dives on a single knob get their own doc (e.g.
[Doubling Decode Speed with MTP](mtp-speculative-decoding.md)); this page is the
flat index across all of them.

- [Log](#log)
- [Image models](#image-models)
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
| gemma-4-31B-it-Q4_K_M | gemma4 (dense, SWA 50/60 layers) | RTX 4090 24 GiB | `ctx_size 81920`, `cache_type_k/v q8_0`, `no_mmproj_offload`, `reasoning true`, no MTP (none baked in) | 42.2 (shallow, no spec) | 22.2 GiB solo / 22.9 GiB with granite embedding co-resident | 2026-08-23 | Dense 31B, no `nextn` tensors in the GGUF (checked both the tensor list and the load log — no "unused tensor" warnings), so `spec_type: draft-mtp` isn't available. `gemma-4-E4B-it-Q6_K` shares its exact 262144-token vocab (verified byte-for-byte) so it's tokenizer-compatible as an external `draft-simple` model, but at 7.1 GiB it doesn't fit alongside the 31B's ~18.2 GiB (weights+mmproj) floor on one 24 GiB card without gutting ctx_size further — not wired up. `calibrate` badly under-recommends here (ctx 2048, false "weights exceed usable VRAM" alarm) — see gap below. `no_mmproj_offload` (the Qwen3.8-27B trick) was required: without it, 65536 already left only 869 MiB free and evicted the embedding service; with it, 81920 leaves ~1.3 GiB solo, ~0.6 GiB with granite co-resident — tight but loads and serves without crashing, same risk tier accepted for Qwen3.8-27B's 131072 config. 98304 (still with `no_mmproj_offload`) OOM's the headroom down to 509 MiB solo — one step too far, reverted. |

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

**2026-08-23 — third calibrate gap, on `gemma4` (Gemma-4-31B):** this arch
*does* publish the per-layer boolean `sliding_window_pattern` `calibrate`
knows how to read, but it still prices SWA layers using the full-attention
`attention.key_length`/`value_length` (512) instead of the
`_swa`-suffixed pair (`key_length_swa`/`value_length_swa` = 256) that this
GGUF also carries. Effect: it estimated the 50 SWA-layer cache at 2.3 GiB
where hand math with the correct dims gives ~0.85 GiB (f16) — badly
oversized — on top of a separate false-positive "weights + reserve exceed
usable VRAM" alarm (18.2 GiB actually fits fine under the 21.1 GiB it
computed as usable). Net effect: recommended `ctx_size: 2048` against a
`ctx_size: 81920` that's actually stable and measured. Don't trust
`calibrate`'s ctx number on `gemma4` SWA models — do the per-layer byte math
by hand (same method as the MTP doc's hybrid-arch formula, just swap in
`key_length_swa`/`value_length_swa` for the SWA-tagged layers) and verify
with a real `zallama load` + `nvidia-smi`. Not fixed yet.

## Image models

Diffusion has no token axis, so these rows report **seconds per image** instead
of decode tok/s. Measured with `zallama bench <model> --image-size WxH`; the
full write-up is [Tuning Image Generation](sd-tuning.md).

| Model | Backend | GPU | Config | s/image @ 1024x1024 | VRAM | Date | Notes |
|---|---|---|---|---|---|---|---|
| FLUX.1-schnell-Q4_0 (`flux:klein`) | sd-server (`master-813`) | RTX 4090 24 GiB | `steps 4`, `sampler euler`, `cfg_scale 1.0`, `diffusion_fa`, `vae_tiling`, `t5xxl_fp16` | 3.95 | 17.0 GiB | 2026-09-02 | Starting config. The fp16 T5 encoder (9.8 GB) cost more than the Q4_0 diffusion weights (6.8 GB), and at 17 GiB the model could not co-exist with anything on the card. |
| FLUX.1-schnell-Q4_0 (`flux:klein`) | sd-server (`master-813`) | RTX 4090 24 GiB | as above but `t5xxl` = `t5-v1_1-xxl-encoder-Q8_0.gguf`, `vae_tiling false` | **3.30** | **12.3 GiB** | 2026-09-02 | Winning config. Q8_0 text encoder is visually lossless and frees 5.2 GiB; dropping tiling buys 20% back. 10.6 GiB now free for a co-resident text model. |
| FLUX.1-schnell-Q4_0 (`flux:klein`) | sd-server (`master-813`) | RTX 4090 24 GiB | `vae_tiling`: true vs false, else as winning config | 3.95 vs 3.30 | 11.8 vs 12.3 GiB | 2026-09-02 | Tiling is a memory trade, not a free one: 0.5 GiB for 20% of the clock. The README's old "~6.6 GB decode buffer" figure does not hold on this build. |
| FLUX.1-schnell-Q4_0 (`flux:klein`) | sd-server (`master-813`) | RTX 4090 24 GiB | `fa` x `diffusion_fa`, else as winning config | 3.31 / 3.32 / 3.32 / 5.72 | 12.3 GiB | 2026-09-02 | Flash attention is worth **1.73x**, and all of it lives in the diffusion model — `fa` on top of `diffusion_fa` measures identical. Keep `diffusion_fa` alone. |
| FLUX.1-schnell-Q4_0 (`flux:klein`) | sd-server (`master-813`) | RTX 4090 24 GiB | `vae_conv_direct` / `diffusion_conv_direct`, else as winning config | 9.01 vs 3.31 | 11.8 vs 12.3 GiB | 2026-09-02 | `vae_conv_direct` buys the same 0.5 GiB as tiling and costs **2.7x** the wall time — never worth it here. `diffusion_conv_direct` is noise in both directions. |
| FLUX.1-schnell-Q4_0 (`flux:klein`) | sd-server (`master-813`) | RTX 4090 24 GiB | `steps`: 4 vs 8 @ 512x512, else as winning config | 0.93 vs 1.54 | 12.3 GiB | 2026-09-02 | Near-linear in steps, as expected — which is also why `cache_mode` has nothing to skip at 4 steps. |

---

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
