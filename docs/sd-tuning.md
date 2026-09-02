# Tuning Image Generation (sd-server)

*Measured on an RTX 4090 24 GiB, `flux:klein` (FLUX.1-schnell Q4_0),
stable-diffusion.cpp `master-813-bfbef5b`, 2026-09-02. Every number here came
out of `zallama bench`, not an estimate.*

Image generation is tuned differently from text. There is no token axis and no
KV cache; the cost is **latent area × steps** for sampling, plus a VAE decode
whose buffer grows with the square of the image size, plus a text-encoder pass
that runs once per prompt. What follows is what actually moved the numbers.

- [The text encoder is the VRAM problem](#the-text-encoder-is-the-vram-problem)
- [What moved the clock](#what-moved-the-clock)
- [What did nothing](#what-did-nothing)
- [Knobs worth reaching for next](#knobs-worth-reaching-for-next)
- [Benchmarking image models](#benchmarking-image-models)

---

## The text encoder is the VRAM problem

The starting stack was 17.0 GiB, and the diffusion model was the smaller half
of it:

| Component | File | Size |
|---|---|---|
| Diffusion | `flux1-schnell-Q4_0.gguf` | 6.77 GB |
| **Text encoder** | **`t5xxl_fp16.safetensors`** | **9.79 GB** |
| CLIP-L | `clip_l.safetensors` | 0.25 GB |
| VAE | `ae.safetensors` | 0.34 GB |

T5-XXL at fp16 costs more than the model it conditions. Swapping it for
[`t5-v1_1-xxl-encoder-Q8_0.gguf`](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)
(5.06 GB) took the resident stack from **17.0 → 11.8 GiB measured**, with no
visible change in prompt adherence — Q8_0 on a text encoder is effectively
lossless.

```yaml
artifacts:
  t5xxl: /bank2/zallama/models/t5-v1_1-xxl-encoder-Q8_0.gguf
mem_gb: 11.8
```

This barely changes generation speed — the encoder runs once per prompt, not
once per step. It matters because of **eviction**. At 17 GiB an image model
cannot co-exist with anything on a 24 GiB card, so every image request evicts a
large text model and both pay a reload. At 11.8 GiB it fits alongside an ~11 GiB
model inside the same `mem_budget_gb`. See
[eviction groups](../CONFIG.md) for how that budget is enforced.

The same trick applies to any stack with a heavyweight conditioner —
`qwen-image:20b` ships a 7B Qwen2.5-VL text encoder on `--llm`.

## What moved the clock

All at 1024×1024, `steps: 4`, `sampler: euler`, `cfg_scale: 1.0`, 2 runs after
a warmup.

| Change | s/image | Δ | VRAM |
|---|---|---|---|
| Flash attention off (`fa`/`diffusion_fa` both false) | 5.72 | baseline | 12.3 GiB |
| **`diffusion_fa: true`** | **3.31** | **1.73× faster** | 12.3 GiB |
| `vae_tiling: true` | 3.95 | 1.20× slower | 11.8 GiB |
| **`vae_tiling: false`** | **3.30** | — | 12.3 GiB |
| `vae_conv_direct: true` | 9.01 | **2.7× slower** | 11.8 GiB |

Two things to take from that table.

**`vae_tiling` is a memory trade, not a free one.** It was switched on to
survive a VAE decode buffer that the README describes as ~6.6 GB on top of the
weights. On this build it is worth **0.5 GiB** and costs **20 %** of the clock,
so once the T5 swap freed room it was pure loss. Re-measure it before enabling
it on a new model rather than assuming the old figure.

**`vae_conv_direct` is a trap.** It buys the same 0.5 GiB as tiling and costs
2.7× the wall time. `diffusion_conv_direct` is free either way (3.31 vs 3.32 s,
inside the noise) — it neither helps nor hurts here.

## What did nothing

- **`fa` on top of `diffusion_fa`.** `fa` enables flash attention everywhere,
  `diffusion_fa` only in the diffusion model. All of the 1.73× lives in the
  diffusion model: `fa=true diffusion_fa=true` (3.31 s) and
  `fa=false diffusion_fa=true` (3.32 s) are the same number. The text encoder
  and VAE have no attention worth accelerating. Keep `diffusion_fa`.
- **`diffusion_conv_direct`.** Noise in both directions.

## Knobs worth reaching for next

`SdServerBackend` passes only the flags it maps and silently drops the rest, so
anything below has to be in `_PARAM_MAP` / `_FLAG_MAP` before `zallama set` can
reach it. These are mapped as of v1.14.0 but not yet measured here:

| Param | Why |
|---|---|
| `cache_mode` (`easycache`, `dbcache`, `taylorseer`, `spectrum`) | Reuses block activations across timesteps — usually the largest decode lever a diffusion model has. **Worth nothing at `steps: 4`**: there is no redundancy to skip. Test it on a 20-step model (`flux:dev`, `qwen-image:20b`). |
| `eager_load` | sd-server answers its health check *before* the weights are resident — they load lazily on first use. `LOAD s` therefore reads 0.5 s while the real cost hides in the first generation. `eager_load` moves it back into startup, where the process manager's health check already waits. |
| `max_vram` + `stream_layers` | Graph-cut segmented execution: run the graph in slices that fit a budget. A negative value auto-detects free VRAM sparing that many GiB, which is the principled way to co-exist with a text model instead of `offload_to_cpu`. |
| `taesd` | Tiny autoencoder, a few MB, decodes far faster than the 335 MB VAE. Its main draw was avoiding tiling — less compelling now that tiling is off, but still the fastest decode path for previews. |
| `hires` + `hires_scale` | Sample at 512 and upscale. Diffusion cost grows with latent area, so this is much cheaper than sampling natively at 1024. |
| `tensor_type_rules` | Per-pattern load-time quantization, e.g. `^vae\.=f16,model\.=q8_0` — the alternative to the T5 swap when no quantized file exists. |

## Benchmarking image models

`zallama bench` measures image models as of v1.14.0. There is no token axis, so
`--prompt-tokens`, `--max-tokens` and `--temp` are ignored; `--concurrency`
still applies.

```bash
zallama bench flux:klein --image-size 1024x1024 --sweep vae_tiling=true,false
zallama bench flux:klein --sweep steps=4,8 --sweep cache_mode=,easycache
zallama bench flux:klein --sweep fa=true,false --runs 3 -o sd.json
```

Two caveats when reading the output:

- **`IMAGE s` is wall time at this client.** sd-server publishes no per-step
  clock the way llama.cpp publishes `timings`, so there is nothing to read off
  the engine — queueing and proxy overhead are included. That is the honest
  number for "how long until I have a picture" anyway.
- **`LOAD s` stops at the health check**, and so excludes the weights unless
  `eager_load` is set. The warmup generation absorbs the real load; the VRAM
  column is re-read after it for the same reason.

An empty sweep value removes the param, so `--sweep cache_mode=,easycache`
compares "off" against "on" in one run.
