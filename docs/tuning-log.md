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
| Qwen3.8-27B-UD-Q3_K_XL | qwen35 (hybrid) | RTX 4090 24 GiB | `ctx_size 196608`, `cache_type_k/v q8_0`, `spec_type draft-mtp`, `spec_draft_n_max 3`, `no_mmproj_offload`, shared `mmproj-BF16-Qwen3.8-27B-Q4_K_M.gguf` | **87.9 ±1.6** honest (43% accept); n_max 1/2/3/4 → 82.9/88.1/87.9/81.9 | 21.2 GiB (2.4 GiB free) | 2026-09-12 | Same arch/MTP as the Q4_K_M row, 3.7 GiB less weights (12.2 GiB) — spent entirely on context: 196k vs the Q4_K_M's 108k. Not faster than Q4_K_M: the smaller quant buys ctx, not tok/s. First measured at 104.5 on the old copy-prompt (see the 2026-09-12 bench note below); 88 is the number on original prose. 262144 (full native) OOM'd exactly at the MTP draft context's 1 GiB KV alloc — the last thing llama-server allocates — so 196608 is one step below the ceiling; `calibrate --probe` with the default 2.5 GiB margin would land a step lower. Static `calibrate` said 102400 (f16 KV, 88 % usable). |
| Qwen3.8-27B-UD-Q5_K_M | qwen35 (hybrid) | RTX 4090 24 GiB | `ctx_size 65536`, else as Q3_K_XL row | **68.2 ±1.5** honest (39% accept); 94.7 on the old copy-prompt | 22.2 GiB (1.9 GiB free) | 2026-09-12 | 18.4 GiB weights, so KV budget is ~2.3 GiB → 64k is the max at 1.9 GiB free; `calibrate --probe` (default margin 2.5 GiB) says 49152, measured slope 42.8 KiB/token. ~22 % slower decode than Q3_K_XL on real text (more bytes per token, lower acceptance). Pick this one for quality at ≤64k, Q3_K_XL for long-context, Q4_K_M for the middle. Static `calibrate` refused it outright ("weights + reserve exceed usable VRAM") — wrong, it fits. |
| Nex-N2.5-mini-Q4_K_S (`nex:n2.5-mini`) | qwen35moe (hybrid MoE, 256 experts/8 active, 10 of 40 layers full-attn, 2 KV-heads) | RTX 4090 24 GiB | `ctx_size 196608`, `cache_type_k/v q8_0`, `no_mmproj_offload`, `reasoning false`, temp 0.7 / top_p 0.95 / top_k 40 (model card), no MTP (no `nextn` in the GGUF) | **192.5 ±0.5** greedy (prompt 468), 5078 tok/s prefill, TTFT 169 ms | 21.9 GiB (2.2 GiB free) | 2026-09-12 | Nex-AGI's agentic/computer-use model on the Qwen3.5-35B-A3B foundation; full GPU offload, no `n_cpu_moe` needed at Q4_K_S (18.5 GiB weights). KV is cheap here (only 10 caching layers × 2 KV-heads): measured slope 13.3 KiB/token, so the full native 262144 *does* fit (22.7 GiB, 1.3 GiB free) — backed off to 196608 for the same ~2 GiB services margin as the 27Bs; bump back to 262144 if it ever runs solo. Tool-calling (OpenAI `tools` → structured `tool_calls`) and vision via the F16 mmproj (on CPU) both verified with real requests. `calibrate` refused it too (same >19 GiB-weights false alarm) and recommended ctx 2048. |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 | nemotron_h_moe (hybrid MoE) | RTX 4090 24 GiB | Re-tune: **no speculation** (`spec_type`/`spec_draft_n_max` removed — was draft-mtp), `ctx_size 450560` (via `calibrate --probe --apply`), `batch_size 2048`, `ubatch_size 2048` (was 512), `cache_type_k/v q8_0`, `reasoning false` (was unset = thinking on: a "17 × 23?" ate all 400 max_tokens in reasoning_content and returned empty content) | **237.2 ±2.2** greedy on original prose; with MTP: n_max 1 → 247.3 (62% accept), 2 → 241/238 (48%), 3 → 221 (38%), 4 → 206 (30%), 6 → 161 (20%); prefill **5321** without MTP vs 4200 with, **8605** vs 7108 on a 3.6k prompt with ubatch 2048 | 21.0 GiB (2.5 GiB free) | 2026-09-12 | Goal: more speed, ctx ≥ 105k. Started the day picking `spec_draft_n_max 4` off 341–356 tok/s sweeps — every one of those was the model **copying the bench filler** (100% draft acceptance, see the bench note below); on text it has to write, MTP buys **+4 % at best** (n_max 1) for +2.1 GiB VRAM, −20 % prefill and +25 ms TTFT, and gets worse with every extra drafted token. Dropped it: A3B active weights are so cheap per token that verifying a draft costs what it saves. Without the draft context the KV slope is 7.9 KiB/token (was ~13 with it), so the full 450k fits with 2.5 GiB spare — more ctx *and* more free VRAM than the August 444416 entry. `ubatch_size` 2048 stays: +21 % prefill for +0.9 GiB, the one clean win. KV f16 vs q8_0: no consistent difference. Tool calling: declines a plain "what's the weather?" but calls `get_weather` fine when told to use it — `probe` prompts imperatively for that reason. |
| Muse-Glimmer-30B-UD-Q4_K_XL (`muse-glimmer:30b`) | muse-glimmer (dense, SWA 39/52 layers) | RTX 4090 24 GiB | Re-tune: `ctx_size 131072` (full native, was 24576), `no_mmproj_offload` (new), `cache_type_k/v f16` (was q8_0), `spec_draft_n_max 2` (was 4), `spec_type draft-dflash` kept, `chat_template_kwargs '{"reasoning_strength":"low"}'` (new) | **82.3 ±2.0** honest @ n_max 2 (49% accept); n_max 1/2/3/4/8/16 → 70.6/82.3/80.3/79.3/67.7/68.7 at 65/49/34/29/16/10% accept. (Old copy-prompt: 107 @ n_max 3 f16, 99 q8_0.) | 18.8 GiB (4.8 GiB free) | 2026-09-12 | Goal: more speed, ctx ≥ 105k. The 24576 cap was never the KV: `sliding_window_pattern 4` → only 13 of 52 layers are full-attention, × 2 KV-heads × head_dim 128 = **~7 KiB/token** (q8_0), so the whole 131072 costs < 1 GiB. It was the **3.6 GiB BF16 mmproj** sitting in VRAM; `no_mmproj_offload` moves it to CPU and the full native ctx fits with ~5 GiB to spare. Real decode is ~80–82 tok/s, up from the 52.8 of August (llama.cpp's dflash path improved); the 107 first written here was copy-speed. DFlash draft length: 2 wins on real prose, 1 is clearly worse (fewer tokens per verify), ≥8 falls off a cliff as acceptance collapses. KV f16 measured +7 % on the copy-prompt; kept, it's 0.7 GiB on a model with this little KV. `ubatch_size` 1024/2048: no prefill gain (dense = compute-bound) and +1.1/+3.3 GiB, left at default. **Reasoning**: Harmony-style template (`<|start|>assistant to=self<|message|>…<|eom|>` then `to=user`); the model *always* opens the `to=self` channel and starts by echoing the user prompt into it, then thinks. `reasoning: false` is a no-op (no `<think>` toggle) — the only lever is the template's own `reasoning_strength` variable (default `high`): `low`/`none`/`minimal` all cut a "17 × 23?" from ~100 to 47 completion tokens; the prompt-echo residue stays. Wired as the `chat_template_kwargs` param. `probe`: answer 391 in 41 tokens, structured tool call, sees red. |
| gemma-4-31B-it-Q4_K_M | gemma4 (dense, SWA 50/60 layers) | RTX 4090 24 GiB | Re-tune: `ctx_size 24576` (was 49152), `mem_gb 20.6` (was 22.2), `spec_type draft-mtp` + `spec_draft_n_max 3` with the external `mtp-gemma-4-31B-it-BF16.gguf` draft (0.9 GiB, added 2026-08-23 after the row above was written — the "no MTP" there is stale), `cache_type_k/v q8_0`, `no_mmproj_offload`, `reasoning false` | ~79 tok/s with MTP (64% accept on an 11k-prompt run in the log; 42 without) | 20.6 GiB (0.9 GiB free with 2.0 GiB of services resident) | 2026-09-12 | **Why it was re-tuned: it stopped loading.** 13 consecutive `died during startup` — every one an OOM on the *last* allocation, the MTP draft context's 520 MiB compute buffer. Nothing about the model changed; `tdt-0.6b-v3-q8_0` (parakeet, 1.4 GiB) had joined granite (0.7 GiB) in the always-resident services group, and the daemon can't evict services for a primary model, so it admitted the 22.2 GB declaration over budget and llama-server hit the wall. `calibrate --probe` measured **60.2 KiB/token all-in** (draft ctx + buffers) on a **20.1 GiB floor at ctx 16384** — so 49152 needs ~22.0 GiB for the model alone, which only ever fit when granite was the sole service. Ceiling next to 2 GiB of services: 24576 (0.9 GiB free), 28672 leaves 0.7. The alternative is dropping MTP (frees the 0.9 GiB draft + its context, ~1.5–2 GiB) to get back to ~48k at 42 tok/s — chose speed over context here: this model is picked for quality, and 79 vs 42 tok/s matters more day-to-day than 24k vs 48k. Flip with `zallama set gemma-4-31B-it-Q4_K_M spec_type= spec_draft_n_max= ctx_size=49152` and re-probe if long context is needed. General lesson: a primary model whose `mem_gb` sits within ~2 GiB of the card only fits until the next service model is registered — re-probe primaries whenever the services group grows. |

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

**2026-09-12 — the "weights + reserve exceed usable VRAM" false alarm is
the dominant calibrate failure now, not a gemma4 quirk.** It fired on both
`Qwen3.8-27B-UD-Q5_K_M` (18.4 GiB weights) and `Nex-N2.5-mini-Q4_K_S`
(18.5 GiB) — any file ≳ 18 GiB trips it, because "usable" is hardcoded to
88 % of the card (21.1 GiB) and a 1.9 GiB compute reserve is stacked on top
before the KV math even starts. Both models load fine with ≥ 1.9 GiB
*physically* free at their measured ctx (rows above). When it fires,
`calibrate` also silently drops the GGUF dims ("size-only estimate") and
recommends `ctx_size: 2048`, so the number is useless rather than merely
conservative. Rule of thumb that matched reality within ~0.3 GiB across all
three models today: `weights + mmproj (0 if no_mmproj_offload) + KV
(bytes/token × ctx, q8_0 = 1.0625 × f16) + ~1.5 GiB compute/draft` should
land ≤ ~22 GiB on the 24 GiB 4090, leaving ~2 GiB for the `services` group.
Then verify with a real load.

**2026-09-12 — `zallama bench` was measuring copy speed on speculative models.**
The bench prompt ended with "Continue the passage above" after ~500 tokens of
repeated filler prose, and models did exactly that: continued by repeating
the filler. An MTP/DFlash draft predicts a verbatim copy perfectly, so
acceptance sat at **100 %** (visible once `bench` grew its `ACCEPT %` column),
every longer `spec_draft_n_max` looked better, and Nemotron "did" 330–470
tok/s. Fixed the prompt to demand an original short story; on that, Nemotron
does 237 tok/s *without* MTP and 247 with it at n_max 1, Qwen3.8-27B ~88
(not 105), Glimmer ~82 (not 107). **Every speculative-decoding decode figure
in this log dated before 2026-09-12 is inflated by this**, the direction of
the `spec_draft_n_max` sweeps included; the rows above from today carry both
numbers. Non-speculative decode figures (Nex 192, Gemma, etc.) are unaffected
— per-token cost doesn't depend on what the token is. Prefill figures are
unaffected. Sweep-reading rule from now on: read `ACCEPT %`; anything near
100 % is the model copying something, not generating.

Tools added the same day so this stops being hand work:
`zallama calibrate <m> --probe [--apply]` (bisect ctx_size by real loads,
default margin = services budget), `zallama probe <m>` (does it think / call
tools / see; which template switch it understands), `bench`'s `ACCEPT %`.

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
