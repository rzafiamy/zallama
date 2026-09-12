# MiniCPM5-2B as an Agentic Model

*Testing OpenBMB's MiniCPM5-2B for tool-calling and multi-step agent loops
against the registered `minicpm5-2b-q4_k_m` on a 4090, 2026-09-10.*

MiniCPM5 is OpenBMB's **text-only** on-device LLM line (`LlamaForCausalLM`,
Apache-2.0, 131072 trained ctx). It is *not* the MiniCPM-**V** multimodal line —
easy to conflate, and the two are unrelated releases. MiniCPM5-1B landed first,
MiniCPM5-2B (2.52B total / 1.98B non-embedding) on 2026-09-07.

- [Why it matters here](#why-it-matters-here)
- [What works](#what-works)
- [Numbers](#numbers)
- [Registry sizing](#registry-sizing)
- [Takeaway](#takeaway)

---

## Why it matters here

The interesting claim is agentic competence at a size that fits *beside*
everything else on the card. Compare [Qwen3-0.6B](qwen3-0.6b-agentic.md), which
handles single-shot tool calls well but collapses multi-step chains into
same-turn parallel calls with hallucinated intermediate arguments. That failure
is what forces a harness to single-step dependent calls by hand.

The HF card warns that tool calling needs SGLang, "which converts the model's
XML-style tool calls to OpenAI-compatible format". **That caveat does not apply
to zallama.** llama.cpp build 10434 ships a dedicated MiniCPM5 chat parser
(`common_chat_params_init_minicpm5`, log line `Using specialized template:
MiniCPM5`), so `tool_calls` come back already in OpenAI shape. No extra flags —
recent llama-server defaults to `--jinja`.

## What works

Tested at `ctx_size: 20000`, f16 KV, via `POST /v1/chat/completions` on the
daemon (port 6767).

- **Parallel tool calls.** Two independent `get_weather` calls (Paris, Tokyo) in
  one response — correct name, valid JSON arguments, empty `content`,
  `finish_reason: tool_calls`.
- **`reasoning_content` is separated cleanly.** The thinking trace lands in its
  own field, not inlined into `content` — a client can drop or fold it without
  string-munging.
- **Sequential dependencies are sequenced correctly — the Qwen3-0.6B failure
  does not reproduce.** Given `get_user_id` + `get_email_by_id` and a request to
  find a user's email from their username, it emitted **exactly one** call
  (`get_user_id{"username":"jdoe"}`) and waited. Fed the result
  (`user_id: 84213`), it then called `get_email_by_id{"user_id":84213}` with the
  *real* id, and after that result produced the correct final answer. A naive
  harness that blindly executes returned `tool_calls` is safe with this model in
  a way it is not with Qwen3-0.6B.
- **Prompt caching across the loop.** Turn 2 came back `cached_tokens: 336`, so
  agent turns only prefill the new tool result.

## Numbers

RTX 4090, `ctx_size: 20000`, f16 KV, Q4_K_M:

| Test | Prompt tokens | Completion tokens | Prompt tok/s | Decode tok/s |
|---|---|---|---|---|
| 2 parallel tool calls (temp 0.3) | 245 | 345 (incl. reasoning) | 6,908 | 328 |

Real VRAM at `ctx_size: 20000`: **2.6 GB** (`nvidia-smi` 2696 MiB with only
`kokoro:82m` otherwise resident).

Note the completion is 345 tokens for a two-call answer — most of it is the
`<think>` trace. This model reasons before every tool call, so wall-clock per
agent step is dominated by reasoning tokens, not by the call itself. Budget for
that, or drive `reasoning` down where the harness supports it.

## Registry sizing

The entry as first pulled was under-specified — no `mem_gb`, and `ctx_size`
truncated to 20000 against a trained 131072:

```yaml
- name: minicpm5-2b-q4_k_m
  file: /bank2/zallama/models/MiniCPM5-2B-Q4_K_M.gguf
  params:
    ctx_size: 20000
```

Missing `mem_gb` means eviction accounting falls back to *GGUF size x 1.2* =
1.9 GB against a measured 2.6 GB — a 27% under-count, exactly the kind of drift
that feeds [eviction thrash](../README.md#-memory-aware-eviction).

`zallama calibrate` reports the arch as `llama • 42L • 2 KV-heads • head_dim
128` — a plain dense GQA model, so calibrate's KV arithmetic is trustworthy here
(unlike the hybrid Qwen3.8 archs, where it under-sizes ctx). It recommends:

```
zallama set minicpm5-2b-q4_k_m ctx_size=131072 mem_gb=7.1
```

7.1 GB is f16 KV at 42 KiB/token. `cache_type_k=q8_0 cache_type_v=q8_0` cuts the
cache ~1.9x, which is the difference between "a model you keep resident" and
"a model that evicts a 21 GB primary". Prefer the q8_0 form unless a full-f16
long-context run is being measured.

## Takeaway

MiniCPM5-2B is the first model at this size to pass the multi-step tool-chain
test that Qwen3-0.6B fails, at ~2.6 GB and ~330 tok/s. That makes it a
credible **default agent brain for zechat** — resident alongside a large primary
model rather than competing with it — where Qwen3-0.6B remains the cheaper
single-shot autocomplete pick.

Two caveats worth keeping honest:

- The "beats Qwen3.5-4B" framing comes from OpenBMB's own 34-benchmark
  comparison table (avg 53.9). The tool-use results above are independently
  reproduced here; the general-capability claim is not.
- It is **text-only**. Nothing about MiniCPM5 replaces the vision path. The
  multimodal sibling is MiniCPM-V 4.6 (1.3B: SigLIP2-400M encoder + Qwen3.5-0.8B
  backbone, 262k ctx, Apache-2.0, llama.cpp support) — a separate pull, and an
  interesting one for the same "small enough to stay co-resident" reason.

---

# 2026-09-10 — Two-model zechat: 2B + 27B co-residency

*Can `minicpm5-2b-q4_k_m` and `Qwen3.8-27B-Q4_K_M` share the 4090, and what
does the split have to look like? Measured, not modelled.*

## VRAM sweep — the 2B alone

`cache_type_k/v: q8_0`, `n_gpu_layers: 99`, via `scripts/measure_vram.py`:

| ctx_size | real VRAM | implied mem_gb |
|---|---|---|
| 8192 | 1.97 GB | 2.1 |
| 16384 | 2.15 GB | 2.3 |
| 32768 | 2.52 GB | 2.7 |
| 65536 | 3.28 GB | 3.4 |
| 131072 | 4.80 GB | 5.0 |

Weights floor ~1.79 GB; q8_0 KV costs ~22 KiB/token, linear. (f16 KV is
42 KiB/token — the 1.9x that `calibrate` advertises.)

## Naive co-residency OOMs

27B at its production `ctx_size: 108000` (21.0 GB) + 2B at 32768 (2.5 GB)
**loads** — 24063 MiB used, 501 MiB free at idle — and then **dies under
concurrent load**. Firing a 400-token 27B completion and a 2B tool call
simultaneously killed the 27B outright:

```
E  cuMemCreate(&handle, reserve_size, &prop, 0)
E  current device: 0, in function alloc at ggml-cuda.cu:589
```

The daemon returned `502 Bad Gateway` for the 27B; `zallama ps` then showed it
with an em-dash for real VRAM (process gone) while still holding its 21.0 GB
budget reservation. **501 MiB of idle headroom is not enough** — the earlier
services-group work settled on 435 MiB as tight-but-survivable for a 1.4 GB
model, and a 27B's compute buffers are a different order of magnitude.

## The fix: buy the 2B's VRAM from the 27B's context

`calibrate` on the 27B reports `qwen35 • 65L (17 with KV) • 4 KV-heads •
head_dim 256`, **36 KiB/token** at q8_0. Only 17 of 65 layers carry KV (hybrid
arch), but 36 KiB/token is still steep:

| 27B ctx_size | KV cost | vs 108000 |
|---|---|---|
| 108000 | 3.89 GB | — |
| 65536 | 2.36 GB | −1.53 GB |
| 32768 | 1.18 GB | **−2.71 GB** |
| 16384 | 0.59 GB | −3.30 GB |

Dropping the 27B from 108000 to 32768 frees **2.71 GB** — almost exactly the
2.52 GB the 2B costs at the same ctx. The two-model split is very nearly
self-financing.

Measured at `Qwen3.8-27B ctx_size=32768 mem_gb=18.4` + `2B ctx_size=32768`:

| | idle free | peak free under concurrent load | 27B decode | 2B decode |
|---|---|---|---|---|
| 27B@108k + 2B@32k | 501 MiB | **OOM — 27B killed** | — | — |
| 27B@32k + 2B@32k | 3703 MiB | **3641 MiB** | 74.3 tok/s | 119.9 tok/s |

Both served concurrently with 3.6 GB to spare.

**Contention is real but acceptable:** the 2B decodes 328 tok/s alone and
119.9 tok/s while the 27B is generating — they share SMs, not just VRAM. Budget
for ~2.7x slowdown on the small model whenever the big one is mid-turn, and do
not design a UI that assumes the 2B is instant during a 27B answer.

## What this justifies architecturally

The split that pays for itself is **the 27B stops being the context holder**.
Its 108k window exists because it is currently the only model in the
conversation; hand the turn-by-turn conversation to the 2B and the 27B only
ever receives a focused reasoning task, which 32k covers comfortably.

Note what is *not* on the table: **the 2B cannot act as a draft model for the
27B.** Speculative decoding requires a shared vocabulary, and MiniCPM5 ships its
own tokenizer against Qwen3.8's. The 27B already uses `spec_type: draft-mtp`
with a baked-in head, which is the better mechanism anyway.

---

# 2026-09-11 — Which orchestration calls should leave the big model?

*Analysis of `zeagentrs` call sites plus a measured router bench, 2B vs 27B.*

## The seam already exists

`zeagentrs` `SessionConfig` carries `router_model: Option<String>`
(`src/agent/mod.rs:299`), and `LLMRouter::new` takes an explicit model:

```rust
Some(Arc::new(LLMRouter::new(
    config.adapter.clone(),
    config.router_model.clone().unwrap_or_else(|| config.model.clone()),
    ...
```

Note `config.adapter.clone()` — the router reuses the **same** adapter. Against
zallama that is exactly right: one `base_url`, different model names, both
resident. A second model costs no adapter work at all.

**zechat never sets `router_model`.** It sets only `router_max_tokens`
(`zechat/src-tauri/src/agent.rs:757`), so routing runs on the 27B today.

## Every LLM call site in the runtime

| Site | Function | What it decides | Model today |
|---|---|---|---|
| `execution/router.rs:561` | `route` | mode + categories + skills | `router_model` ?? `config.model` |
| `agent/mod.rs:1892` | `run_mini_planning_step` | sub-goals for the turn | `config.model` (fixed) |
| `agent/mod.rs:1994` | `reconcile_sub_goals` | goal progress | `config.model` (fixed) |
| `agent/mod.rs:2122` | `verify_goal` | is the goal satisfied | `config.model` (fixed) |
| `agent/mod.rs:3169` | `node_think` | **the answer + tool calls** | `config.model` (fixed) |
| `memory/mod.rs:480` | `select_for_query` | which memories are relevant | own `model` field |
| `memory/mod.rs:703` | `reflect` | end-of-turn memory extraction | `with_adapter(.., model)` |
| `tools/builtin.rs:775` | `SummarizeSandwichTool` | compress bulky tool output | `params.model` ?? `ctx.model` |

Only `node_think` is the product. Everything else is plumbing — which is what
makes the question worth asking.

## Measured: the router, 2B vs 27B

Eight cases through the **real** `router.system.md` / `router.user.md` contract
(five categories, two skills, a nine-tool always-on floor), `temperature: 0`.
Cases: factual, single-category, multi-category, pronoun follow-up,
memory-driven preference, always-on-tool-only, skill-only, prompt injection.

| Config | Correct | Wall (8 calls) | Completion tokens | Parse failures |
|---|---|---|---|---|
| **27B, `reasoning: false`** | **8/8** | 6.8 s | 454 | 0 |
| 2B, reasoning on, free-form | 6/8 | 28.1 s | 8089 | 1 |
| 2B, reasoning off, free-form | 5/8 | 1.8 s | 473 | 1 |
| **2B, reasoning off + JSON schema** | 6/8 | **2.0 s** | 508 | **0** |
| 2B, reasoning on + JSON schema | 6/8 | 32.3 s | 8927 | 1 |

Three things fall out.

**Reasoning is the dominant cost, not parameter count.** The 27B is cheap at
routing *because* its registry sets `reasoning: false` — 454 tokens for eight
decisions. The 2B with thinking on spends 8089 and is 4x slower than the model
thirteen times its size. Any model doing router duty needs reasoning off.

**Most 2B "routing errors" were serialization errors.** Free-form, it emitted
`"categories": "[\"weather\"]"` — a JSON *string* holding a JSON array. The
decision was right; the shape was not. `Vec<String>` will not deserialize from
that, so the runtime would see a routing failure. Constrained decoding
(`response_format: json_schema`, with `categories`/`skills` as enums over the
real catalogue) eliminated every shape failure and took 5/8 to 6/8.

**It is still not good enough to take the router.** The two survivors are the
two the prompt itself warns about: the pronoun follow-up came back with no
categories at all — precisely the "answering 'no category, it refers to a
previous turn' strips the very tools that turn needs" failure the system prompt
was written to prevent — and the skill-only turn put `writing` in `categories`.
Trading 8/8 for 6/8 to save 0.6 s per turn is a bad trade.

## Where the split actually pays

Route by **judgment density**, not by token volume. A call whose output the
agent's own control flow depends on stays on the 27B; a call that compresses or
extracts text does not.

**Move to the 2B:**
- `SummarizeSandwichTool` — the clearest win. High token volume, no judgment,
  and it *already* takes a `model` param, so it needs a caller change, not a
  runtime change.
- `memory::reflect` — end-of-turn, off the critical path entirely. Latency there
  is invisible to the user, and it already has its own model slot.
- `memory::select_for_query` — id-picking from a compact index, own model field.

**Keep on the 27B:** `node_think`, `verify_goal`, `reconcile_sub_goals`,
`run_mini_planning_step`, and — on this evidence — `route`.

## Before moving the router, measure it properly

Eight cases is a direction, not a verdict, and zechat already has the right
instrument: `tests/agentic-bench.mjs`, which sweeps `settings.json` overlays on
a copy of `~/.zechat` and is explicitly "how a tuning claim gets settled instead
of argued". The prerequisite work is worth doing regardless:

1. Add `response_format` / JSON-schema support to the router call, with
   `categories` and `skills` as enums over the live catalogue. This helps the
   27B too (it removes a class of failure rather than trading accuracy).
2. Make the deserializer tolerant of a double-encoded array — one `string-or-seq`
   deserializer turns a hard routing failure into a correct decision.
3. Then sweep `router_model` across 2B and 27B on the real bench.

If (1) and (2) close the follow-up gap, the router moves and the 27B is left
doing only `node_think`. If they do not, the router stays and the split is still
worth having for summarization and reflection alone.

---

# 2026-09-11 — CPU offload and speculation on the 2B: both rejected

*Can the 2B be made smaller in VRAM (CPU offload) and have the lost speed bought
back with speculative decoding? Measured. Neither half works.*

## CPU offload: the worst VRAM per tok/s on offer

`ctx_size: 32768`, q8_0 KV, 42-layer dense model, RTX 4090:

| n_gpu_layers | VRAM | decode | vs full GPU |
|---|---|---|---|
| 99 (all) | 2590 MiB | **303.9 tok/s** | — |
| 32 | 2124 MiB | 99.3 tok/s | −466 MiB, **−67%** |
| 24 | 1776 MiB | 67.7 tok/s | −814 MiB, −78% |
| 16 | 1426 MiB | 50.9 tok/s | −1164 MiB, −83% |
| 8 | 1072 MiB | 41.3 tok/s | −1518 MiB, −86% |
| 0 (all CPU) | 502 MiB | 31.4 tok/s | −2088 MiB, −90% |

The cliff is immediate: the *first* ten offloaded layers cost two thirds of the
speed for 18% of the memory. Do not reach for `n_gpu_layers` on a model this
size.

Why it behaves so much worse than `n_cpu_moe` on a big MoE: `n_cpu_moe` strands
only the **expert** weights in system RAM and keeps attention and the KV cache on
the GPU, so the per-token GPU work is untouched. Plain layer offload moves
everything in that layer, so every token crosses PCIe. MiniCPM5-2B is dense —
there are no experts to strand, and no cheap version of this knob exists for it.

## MTP: there is no baked-in head

`grep -ci nextn` over the model's log returns **0** — no ignored `nextn` tensors,
and `calibrate`'s `llama • 42L` is the real layer count, not 41+1. Unlike
Qwen3.8-27B, MiniCPM5-2B-Q4_K_M ships no MTP head, so `spec_type: draft-mtp` has
nothing to activate. See [MTP](mtp-speculative-decoding.md) for the detection
method.

## DSpark: an SGLang mechanism, not a llama.cpp draft model

OpenBMB does ship a draft model — `openbmb/MiniCPM5-2B-DSpark-GGUF`
(`MiniCPM5-2.6B-DSpark.gguf`, 652 MB) — wired up as a `draft` artifact with
`spec_type: draft-simple`:

| Config | task | decode | draft acceptance |
|---|---|---|---|
| no speculation | counting | **303.9 tok/s** | — |
| `spec_draft_n_max: 4` | counting | 116.2 tok/s | 0.058, mean len 1.23 |
| `spec_draft_n_max: 2` | prose | 145.9 tok/s | 0.074, mean len 1.15 |
| `spec_draft_n_max: 4` | prose | 105.1 tok/s | 0.034, mean len 1.14 |

**3–7% acceptance at every setting**, on both a highly predictable task
(counting) and ordinary prose — so the draft is wrong almost every token and its
cost is paid anyway. Net effect is a 2–3x *slowdown*, plus 652 MB more VRAM.

The log says why:

```
W srv load_model: [spec] failed to measure draft model memory:
    failed to create llama_context from model
```

The HF card describes DSpark as accelerating decoding **in SGLang**. It is a
speculative *architecture* (hidden-state-coupled, EAGLE-shaped), not a small
standalone model of the same vocabulary. llama.cpp loads it as a plain draft and
gets noise. Nothing to tune here — the mechanism is absent, not misconfigured.

## The actual lesson: cut context, never layers

Ranked by VRAM freed per tok/s surrendered, for the two-model layout:

| Lever | VRAM freed | Speed cost |
|---|---|---|
| **27B `ctx_size` 108000 → 32768** | **2710 MiB** | **none** |
| 2B `ctx_size` 32768 → 8192 | 550 MiB | none |
| 2B `n_gpu_layers` 99 → 32 | 466 MiB | −67% |
| 2B + DSpark draft | **−652 MiB** (worse) | −62% |

Context is nearly free to cut until the moment the window is actually needed;
layers are catastrophic from the first one. Both context levers together free
3.2 GB at zero throughput cost, which is more than full CPU offload of the 2B
(2088 MiB) while keeping all 303.9 tok/s.

Speculative decoding remains the right tool where the head is already baked in —
Qwen3.8-27B's `draft-mtp` at 2.2x — and is not available on this model at all.

---

# 2026-09-11 (later) — CORRECTION: DSpark does work

The DSpark conclusion above is **wrong**, and the error was mine: this build
accepts `--spec-type draft-dspark`, and the test used `draft-simple`. DSpark is a
DFlash-derived semi-autoregressive drafter (`block_size=7`, `mask_token_id=75982`),
so `draft-simple` ran it through the wrong path and produced 3–7% acceptance.

Correctly configured, at `ctx_size: 8192`:

| spec_type | VRAM | counting | prose | JSON array |
|---|---|---|---|---|
| `none` | 2030 MiB | 303.9 | 304.0 | 304.7 |
| `draft-dspark` n_max 4 | 3212 MiB | 500.2 | 321.5 | 618.0 |
| `draft-dspark` **n_max 6** | 3212 MiB | 512.9 | 296.9 | **739.0 (2.42x)** |
| `draft-dspark` n_max 8 | 3212 MiB | 491.6 | 280.2 | 732.3 |

Acceptance is 0.90 on JSON, 0.61 on counting, 0.23 on prose at n_max 6: **a
structured-output accelerator**, neutral-to-negative on conversational text.
Also tested and flat: every `ngram-*` mode (294–304 tok/s, no draft model, no
VRAM cost).

Full analysis, plus the `-ot` sweep, the shared-KV-cache question and the
llama.cpp #25618 determinism reproduction, is in
[two-model-agent-study.md](two-model-agent-study.md).
