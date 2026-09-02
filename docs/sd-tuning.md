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

**`vae_tiling` is a memory trade, not a free one — but measure the *peak*, not
the resident size.** The VRAM column above is steady-state residency, where
tiling is worth only 0.5 GiB and costs 20 % of the clock. That column hides the
number that actually decides whether you OOM: the decode buffer is allocated
*after* everything else is resident, and it scales with image area. Peak GPU
usage during one 1024x1024 generation, measured with `nvidia-smi` polling:

| | resident | **peak during decode** | s/image |
|---|---|---|---|
| `vae_tiling: false` | 11.9 GiB | **19.5 GiB** (+6.2) | 3.46 |
| `vae_tiling: true` | 11.4 GiB | **13.4 GiB** (+1.5) | 4.14 |

So the README's ~6.6 GB warning is right, and it applies at 1024x1024 — the
untiled buffer is ~6.2 GiB here. With the whole card to itself Klein has room
for it and tiling is pure loss; the moment anything else is resident, or the
image gets larger, tiling is what keeps the last step from failing. **Re-measure
at the resolution you actually serve**: at 512x512 the untiled buffer is only
~2.7 GiB and the whole question is moot.

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
| `eager_load` | **Measured 2026-09-02: apply it.** sd-server answers its health check *before* the weights are resident — they load lazily on first use, so `LOAD s` reads 0.5 s while ~2.3 s of `loading tensors` hides in the first generation (3.2 s cold vs 0.8 s warm at 512x512). `eager_load: true` moves it into startup, where the process manager's health check already waits: `zallama load` takes 2.7 s and the *first* image costs the same as every later one. Worth it on any model that gets evicted and respawned regularly. |
| `max_vram` + `stream_layers` | Graph-cut segmented execution: run the graph in slices that fit a budget. A negative value auto-detects free VRAM sparing that many GiB, which is the principled way to co-exist with a text model instead of `offload_to_cpu`. |
| `taesd` | Tiny autoencoder, a few MB, decodes far faster than the 335 MB VAE. Its main draw was avoiding tiling — less compelling now that tiling is off, but still the fastest decode path for previews. |
| `hires` + `hires_scale` | Sample at 512 and upscale. Diffusion cost grows with latent area, so this is much cheaper than sampling natively at 1024. |
| `tensor_type_rules` | Per-pattern load-time quantization, e.g. `^vae\.=f16,model\.=q8_0` — the alternative to the T5 swap when no quantized file exists. |

## Step count is the only real speed knob on a distilled model

Klein is already step-distilled, so there is no cache or LoRA left to win with
(`cache_mode` needs redundancy across timesteps; a Turbo/Hyper LoRA re-distils a
model that is already distilled). What remains is the step count itself, and it
is close to linear — measured at 1024x1024, warm, `diffusion_fa: true`,
`vae_tiling: false`, wall time end-to-end through `zallama generate`:

| `steps` | s/image | quality |
|---|---|---|
| 1 | **1.53** | usable; softer background, noisier texture, looser anatomy |
| 2 | **2.15** | very close to 4 — the sweet spot |
| 4 (default) | 3.44 | reference |

That is ~0.65 s per step plus ~0.85 s fixed (VAE decode 0.40 s + HTTP and PNG
encode). Sub-second at 1024x1024 therefore needs `steps: 1` *and* a cheaper
decode (`taesd`), or `hires` — sampling at 512 and upscaling, since diffusion
cost follows latent area.

**Gotcha: sd-server ignores per-request `steps` and `cfg_scale`.** The daemon
forwards them in the request body (`/v1/images/generations` fills unset ones
from the registry), but the backend uses the values it was launched with, so
`zallama generate --steps 1` silently produces a 4-step image at the 4-step
price. Verified against sd-server directly with `steps`, `sample_steps` and
`num_inference_steps` — all three ignored. To change the step count you must
`zallama set flux:klein steps=N` and reload the instance.

**Gotcha: `zallama generate` takes `--size WxH`, and silently drops flags it
doesn't know.** `--width 1024 --height 1024` parses as nothing at all and you
get a 512x512 image at 512x512 speed — which looks like a spectacular result
until you check the PNG header. Always confirm what you actually rendered:

```
python3 -c "import struct;d=open('output.png','rb').read(33);print(*struct.unpack('>II',d[16:24]))"
```

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
