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

### Alternation thrash: two big models that don't fit together

If two `primary` models (two 27B text models, say) each cost more than half your
budget, they can never be co-resident. When clients alternate between them —
even just a chatbot UI with a model switcher, or two agents on the same box —
**every request evicts the other model and cold-reloads a backend**: a 4–8 s
respawn, a burst of `503`s, and the whole cycle repeats on the next request to
the other model. The daemon itself never restarts (`systemctl status zallama`
stays `active (running)`), but `journalctl -u zallama` fills with:

```
Capacity (group 'primary' ...) reached — evicting LRU model 'A' (19.5GB) to make room for incoming 21.0GB
Spawning llama-server for 'B' ...
Capacity (group 'primary' ...) reached — evicting LRU model 'B' (21.0GB) to make room for incoming 19.5GB
Spawning llama-server for 'A' ...
```

There is no config that makes two oversized models coexist. The fix is one of:

- **Route traffic to one model.** Pick the winner and point every client at it.
- **Shrink one** (lower `ctx_size`, quantize KV, offload MoE experts — see
  [Making a model fit](#making-a-model-fit)) until both fit under `mem_budget_gb`
  at once, then raise the budget so neither evicts the other.
- **Accept the reload cost** if the alternation is rare (a few times a day).

What Zallama *does* do automatically: `evict_drain_timeout` (default 30 s) stops
eviction from killing a backend that's still streaming a response. Eviction
prefers an idle victim, and when the only candidate is busy it waits for that
backend to go idle — up to the timeout — before killing it, so an alternating
request no longer truncates the in-flight one with a `502`. It does not stop the
thrash, only its sharpest symptom. Set it to `0` to restore the old
evict-immediately behavior.

Watch out for the **three-way** version of this, which is easy to miss because
each pair looks plausible on its own. An app that uses a big text model, a small
one, and an image model hits it on the RTX 4090: `Qwen3.8-27B` (21.0) + `flux:klein`
(12.3) = 33.3, 27B + `Qwen3.5-4B` (9.5) = 30.5 — only the 4B + flux pair (21.8 with
an embedding model) fits under a 23.8 budget. One oversized member is enough to
make *every* combination thrash, so the diagnosis is per-set, not per-pair. Count
respawns rather than eyeballing the log:

```
journalctl -u zallama --since "-20 min" | grep -oP "Spawning \S+ for '\K[^']+" | sort | uniq -c | sort -rn
```

Anything in double digits over 20 minutes is thrashing. Note that idle unloads
look identical in `zallama ps` (the model is simply gone) but are *not* this:
`idle_timeout` only fires after its full window with no requests at all, and logs
nothing about capacity.

### Editing `pinned` on a model that's already running does nothing yet

`_make_room_locked()` checks `inst.entry["pinned"]` on the **already-running
instance**, not a fresh registry lookup — that entry is a snapshot taken when
the instance was started. Flip `pinned: true → false` (or vice versa) in
`registry.yaml` and the live process keeps behaving on the old value until you
`zallama reload <name>` (or `unload` + `load`) it. Symptom: you unpin a model
to let it share an LRU slot with another, but the *other* model — not the one
you unpinned — keeps getting evicted instead, because eviction skips anything
whose live snapshot still says `pinned: true`. Same family of bug as
`list_models()` serving stale data after a hand-edit — see `zallama set`'s
`mem_gb` placement bug in the tuning log.

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
curl -s -X POST localhost:11436/api/models/<name>/unload   # admin port
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

### Variant: two small services that don't both need to be instant

Adding a third small service (an embedding model, for RAG) doesn't always mean
pinning all three. `Qwen3-Embedding-0.6B-Q8_0`, measured the same way, costs
**2.19 GB** — file size is 0.64 GB, but the KV cache and compute buffer at
`ctx_size: 8192` triple it. Pinning it alongside an already-pinned ASR model
next to a 21 GB text model doesn't fit: 21.0 + 1.3 + 2.3 = 24.6 GB, over a
24.09 GB card.

If ASR and embedding calls aren't both latency-critical in the same instant
(e.g. transcribe-then-generate, or embed-at-ingestion-time rather than
per-turn), leave **both unpinned** instead of pinning either. They then share
one LRU slot and evict each other on demand — cold-load penalty only on
whichever one wasn't just used — while the large text model keeps its own
slot. `max_loaded_models` doesn't need to grow to fit a third always-on
service; it's still `pinned services + concurrent big models`, just with ASR
and embedding no longer counted as pinned.

The failure mode to watch for: if your app calls ASR, embedding, *and* text
generation once per turn, and only 2 non-pinned slots exist, all three compete
and the text model gets evicted and cold-reloaded every turn. That only shows
up under truly interleaved per-turn usage — watch `zallama ps`'s `UPTIME`
column on the text model; if it never exceeds a few seconds, that's the
thrash. The fix then is to give the text model its own reserved slot (pin it,
or raise `max_loaded_models` and accept the larger VRAM footprint), not to
re-pin ASR or embedding.

---

### Variant: a text model and an image model resident together

Text and image models share the `primary` evict group (see *The three
controls*), so a diffusion model loading will evict the text model — and vice
versa — the moment their **declared** `mem_gb` don't both fit under
`mem_budget_gb`. There is no "coexist if there is room" switch: making them
coexist *is* making the declared numbers fit.

Measured 2026-09-02 on the RTX 4090 (24.56 GB card, `mem_budget_gb: 23.8`),
with `granite-embedding-311m` (0.7 GB) also resident:

| `Qwen3.5-4B-Q6_K` `ctx_size` | 4B real VRAM | + `flux:klein` (11.8 GB) + granite | verdict |
|---|---|---|---|
| 262144 (solo default) | 13.0 GB | 25.5 GB | **never fits** — over the card, not just the budget |
| **131072** | 8.8 GB | 21.3 GB steady | fits, but see the peak note below |
| 65536 | 6.7 GB | 19.2 GB steady | fits comfortably |

The 4B's KV cache costs ~40 KB/token at f16, so context is the whole story:
every halving of `ctx_size` gives back ~2 GB. `cache_type_k/v: q8_0` would buy
roughly another halving of the KV portion if more room is needed.

**Decide whether co-residency is worth what it costs.** It is not free: the text
model gives up half or more of its context, and the image model needs tiled VAE
decode. Left alone, each model gets the whole card — the 4B keeps `ctx_size:
262144` (12.9 GB) and FLUX Klein decodes untiled (11.9 GB resident, **19.5 GB
peak** at 1024x1024, 3.46 s per image). The price of *not* co-residing is one
cold reload (4-8 s) each time traffic switches modality. If image requests are
occasional, paying that reload is usually the better trade than permanently
halving the text model's context — co-residency only wins when both are hit in
the same breath, often enough that the reloads dominate.

Two things specific to the image side:

- **`vae_tiling: true` is mandatory when co-resident, and steady state does not
  tell you that.** The VAE decode buffer is allocated *after* everything else is
  already on the card and scales with image area: at 1024×1024 untiled it adds
  **6.2 GB** on top of the 11.9 GB resident footprint (19.5 GB peak), which turns
  a comfortable-looking 21.3 GB steady state into an OOM at the very last step.
  Tiled, the same decode peaks 1.5 GB above resident (13.4 GB) and costs 20 % of
  the clock. At 512×512 the untiled buffer is only ~2.7 GB and none of this
  bites — **a co-residency plan validated at 512×512 is not validated at all**
  (the steady-state rows above were measured that way; re-check the peak at the
  resolution you serve). See `docs/sd-tuning.md`.
- **`sd-server` allocates lazily.** Right after `zallama load flux:klein`,
  `zallama ps` shows `—` for its real VRAM and `nvidia-smi` sees nothing: the
  weights land on the card during the *first generation*, not at load. Verify
  co-residency by generating an image, never by reading `ps` after the load.
  (`eager_load: true` moves that cost into the load if you'd rather.)

**Gotcha: `mem_gb` on a running instance is the value it was started with.**
The eviction math uses `inst.mem_gb`, captured at spawn time — so
`zallama set <model> mem_gb=…` does *not* change what a currently-loaded
instance counts for. Editing the 4B's `mem_gb` from a stale `13.3` down to a
measured `7.0` and then loading the image model still evicted it, because the
running instance was still being counted at 13.3. `zallama reload <model>`
(or `unload` + `load`) after changing `mem_gb`, before loading the other
model. Same rule as `ctx_size` and every other param.

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
