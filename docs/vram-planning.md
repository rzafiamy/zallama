# Fitting Your Models on One GPU

*How Zallama decides what stays loaded, why the default estimate lies to you, and which knob to turn when a model doesn't fit.*

You have one GPU and more models than fit on it. Zallama's job is to decide which
ones are resident at any moment. This page explains how that decision is made,
how to give it accurate numbers, and what to do when a model is too big.

- [The three controls](#the-three-controls)
- [Why the default cost estimate is wrong](#why-the-default-cost-estimate-is-wrong)
- [Measuring what a model actually costs](#measuring-what-a-model-actually-costs)
- [Making a model fit](#making-a-model-fit)
- [A worked example](#a-worked-example)
- [Checklist](#checklist)

---

## The three controls

Three settings decide residency. They are checked in `_make_room_locked()`
before every model start, and eviction always targets the **least recently
used** instance first.

| Setting | Where | What it does |
|---|---|---|
| `max_loaded_models` | `config.yaml` | Hard cap on the **number** of loaded models, all backends counted together. `0` = unlimited. |
| `mem_budget_gb` | `config.yaml` | Cap on the **sum of `mem_gb`** across loaded models. `0` = unlimited. |
| `pinned` | `registry.yaml` | Per model. Pre-loaded at daemon startup and **never evicted**. |

### The trap: pinned models still occupy a slot

`max_loaded_models` counts *every* instance, pinned ones included. If you pin a
TTS model and set `max_loaded_models: 1`, that single slot is permanently taken
and the cap fires on every single model start — you get one pinned model plus
one other, with the other thrashing on every alternation.

**Size the count for what you actually want resident:**

```
max_loaded_models = (number of pinned services) + (number of concurrent big models)
```

Two pinned services (ASR + TTS) and one LLM at a time → `max_loaded_models: 3`.

### When nothing is evictable

If the cap is reached and every loaded model is pinned, Zallama logs a warning
and **admits the incoming model anyway**, over budget:

```
Capacity reached but all loaded models are pinned — admitting incoming 23.4GB over budget.
```

The budget is a scheduling hint, not a hard gate. It will not save you from an
OOM — that is what accurate `mem_gb` values are for.

### What `pinned` is for

Pinning trades memory held for the process lifetime against a cold load you
never pay again. It is worth it for **small, latency-sensitive, frequently
alternating** models — ASR and TTS especially, where a 1 GB model would
otherwise be evicted by every chat request and reloaded on the next
transcription.

```yaml
- name: tdt-0.6b-v3-q8_0
  modality: asr
  backend: parakeet-server
  mem_gb: 1.3
  pinned: true      # pre-warmed at startup, never evicted
```

Never pin a model that would leave no room for your largest one — see the
worked example below.

---

## Why the default cost estimate is wrong

When a model has no `mem_gb`, Zallama estimates it (`_estimate_cost()`):

```python
size_gb = model_path.stat().st_size / 1e9
return round(size_gb * 1.2, 2)          # GGUF size + 20%
```

That heuristic ignores three things that often dominate:

1. **The KV cache**, which scales with `ctx_size`. At long contexts it can be
   several times the weights.
2. **Artifacts** — `mmproj` for vision, and for image models the text encoders
   (`t5xxl`, `clip_l`) and VAE. None are counted.
3. **Compute buffers**, which grow with batch size.

Measured on one machine (RTX 4090, 24.09 GB):

| Model | Estimated | Measured | Error |
|---|---:|---:|---|
| `Qwen3.5-4B-Q6_K` (ctx 262144, KV f16) | 4.23 | **12.92** | +205 % |
| `Qwen3.5-4B-MTP-Q6_K` (ctx 262144, KV q8_0) | 4.37 | **10.91** | +150 % |
| FLUX + t5xxl + clip + vae | 8.12 | **16.20** | +100 % |
| `gemma-4-E4B-it-Q6_K` | 8.49 | **7.60** | −10 % |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | 26.56 | **23.44** | −12 % |

The errors go both ways, so a safety margin on the budget does not rescue you.
**Enabling `mem_budget_gb` on top of unmeasured values is worse than leaving it
off**: it will evict models that would have fit and admit models that will OOM.
Measure first, then enable the budget.

---

## Measuring what a model actually costs

```bash
python3 scripts/measure_vram.py                      # every model in the registry
python3 scripts/measure_vram.py Qwen3.6-27B-MTP      # just these
python3 scripts/measure_vram.py --write              # write mem_gb into registry.yaml
```

The script builds each model's command line through Zallama's own
`backend.build_args()` — so it measures exactly the process the daemon would
spawn — launches it outside the daemon on a free port, waits for `/health`,
reads the per-PID VRAM from `nvidia-smi`, and kills it.

**Run it on an idle GPU.** Another resident model will either skew the reading
or make large models fail to allocate their KV cache. Stop the daemon, or
unload everything first:

```bash
curl -s -X POST localhost:11435/api/models/<name>/unload
nvidia-smi --query-gpu=memory.free --format=csv   # confirm the card is empty
```

Two backends need care:

- **`sd-server` (image)** allocates nothing at load — the weights come up on the
  first generation. The script runs a real 512×512 generation to catch the peak.
  If you normally generate at 1024×1024, raise the value it reports.
- **`kokoro-server` (TTS)** is ONNX on CPU. It uses no VRAM; leave its `mem_gb`
  at `0` and let it be pinned for free.

Record the result with a margin of ~0.1–0.3 GB:

```yaml
- name: Qwen3.6-27B-MTP-Q4_K_M
  mem_gb: 22.2      # measured 22.08
```

---

## Making a model fit

When a model exceeds what the card can give it, work down this list. The order
matters: the first two are free, the rest cost you something.

### 1. Quantize the KV cache — usually free

```yaml
params:
  cache_type_k: q8_0
  cache_type_v: q8_0
```

Roughly halves KV memory at a quality cost most workloads never notice. On a 4B
model at `ctx_size: 262144` this was the difference between **12.92 GB** (f16)
and **10.91 GB** (q8_0) — and the q8_0 variant was carrying an extra MTP draft
head. If a long-context model has no `cache_type_*`, this is your first move.

### 2. Lower `ctx_size` — free if you don't need the context

KV cost is linear in context. But check the weights first: **on a large model
the floor is the weights, and cutting context buys almost nothing.**

`Qwen3.6-35B-A3B-UD-Q4_K_M`, whose weights + mmproj alone are 23.0 GB:

| `ctx_size` | VRAM |
|---:|---:|
| 131072 | 23.44 |
| 98304 | 23.02 |
| 16384 | 21.95 |

Dropping three quarters of the context saved 1.5 GB. Do the arithmetic before
sacrificing context.

### 3. Offload MoE experts to RAM — the best lever for MoE models

For Mixture-of-Experts models (`A3B`, `A4B` in the name), `n_cpu_moe` keeps the
expert weights of the first N layers in system RAM while attention and the KV
cache stay on the GPU:

```yaml
params:
  n_cpu_moe: 4
```

Same 35B model, full 131072 context, mmproj kept:

| `n_cpu_moe` | VRAM | tok/s |
|---:|---:|---:|
| 0 | 23.44 | 177.9 |
| 2 | 22.68 | 154.2 |
| **4** | **21.77** | **137.7** |
| 8 | 19.96 | 113.0 |

About **0.4 GB freed and ~8 % of generation speed lost per layer**. Four layers
bought 1.67 GB for −23 % speed — while keeping the full context *and* vision,
neither of which the other levers could preserve. Needs enough system RAM to
hold the offloaded experts.

Useless on dense models: there are no experts to move.

### 4. Drop the `mmproj` — costs you vision

Removing the `artifacts.mmproj` entry frees the projector's VRAM (0.6–1.2 GB
typically). Reasonable when a *smaller* model in your registry already covers
vision. `--no-mmproj-offload` keeps vision by putting the projector on CPU, at
the price of slower image encoding.

### 5. Lower `n_gpu_layers` — last resort on dense models

Moves whole transformer layers to CPU. Effective but blunt: on a dense model
every offloaded layer is read from RAM on every token, so throughput falls much
faster than with `n_cpu_moe`.

---

## A worked example

One RTX 4090 (24.09 GB usable), a 22 GB MoE model, and an ASR model that should
answer instantly.

**Goal.** Keep speech-to-text warm at all times, and still be able to run the
largest LLM.

**Step 1 — measure.** `tdt-0.6b-v3-q8_0` costs 1.25 GB.
`Qwen3.6-35B-A3B-UD-Q4_K_M` costs 23.44 GB.

**Step 2 — the conflict.** 23.44 + 1.25 = 24.69 GB on a 24.09 GB card. Pinning
the ASR model makes the largest LLM unloadable.

**Step 3 — pick a lever.** Cutting context to 16384 would work (21.95 + 1.25 =
23.20) but costs seven eighths of the context. Dropping the mmproj would work
too (21.93) but costs vision. `n_cpu_moe: 4` costs 23 % of generation speed and
**keeps both**. That is the one to take.

**Step 4 — the config.**

```yaml
# config.yaml
llama_server:
  idle_timeout: 1800
  max_loaded_models: 3      # 2 pinned services + 1 big model
  mem_budget_gb: 23.5       # only meaningful because every mem_gb below is measured
```

```yaml
# registry.yaml
- name: tdt-0.6b-v3-q8_0
  modality: asr
  backend: parakeet-server
  mem_gb: 1.3               # measured 1.25
  pinned: true

- name: Qwen3.6-35B-A3B-UD-Q4_K_M
  artifacts:
    mmproj: mmproj-BF16-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  mem_gb: 21.9              # measured 21.77 with n_cpu_moe=4
  params:
    ctx_size: 131072
    cache_type_k: q8_0
    cache_type_v: q8_0
    n_cpu_moe: 4
```

**Result.** Every model in the registry now coexists with the pinned ASR, the
largest with 0.89 GB to spare. Transcription latency drops from ~780 ms
(reload + inference) to ~80 ms, because the ASR model never leaves the GPU.

---

## Checklist

1. Stop or drain the daemon so the GPU is idle.
2. `python3 scripts/measure_vram.py --write` — real `mem_gb` on every entry.
3. Pin the small always-on services (ASR, TTS). Nothing large.
4. `max_loaded_models` = pinned count + how many big models you want at once.
5. Check the largest model still fits *alongside* the pinned ones. If not, apply
   a lever from [Making a model fit](#making-a-model-fit) and re-measure.
6. Only now set `mem_budget_gb`, a little under your card's usable VRAM.
7. `systemctl restart zallama` — `config.yaml` is read only at startup.
8. Confirm with `zallama ps`: the pinned models should be up before any request.

> **Caveat.** `mem_gb` is a scheduling number, not an enforced limit. The OS owns
> the memory of each backend subprocess; the budget governs *how many* models
> Zallama keeps resident, and it can be overridden when only pinned models
> remain. Accurate values are what keep you off the OOM path.
