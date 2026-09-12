# Running a Small Model Beside a Big One

*A measured study of MiniCPM5-2B and Qwen3.8-27B sharing one RTX 4090: what a
2B can actually take over from a 27B, what it costs in VRAM, and the three
optimisations that turned out not to work. Worked through on zallama v1.14.0,
llama.cpp build 10434, 2026-09-10/11.*

- [The question](#the-question)
- [First, a naming correction](#first-a-naming-correction)
- [Part 1 — Is the 2B actually agentic?](#part-1--is-the-2b-actually-agentic)
- [Part 2 — Can both models stay resident?](#part-2--can-both-models-stay-resident)
- [Part 3 — Which work should move?](#part-3--which-work-should-move)
- [Part 4 — Speculation and the VRAM levers](#part-4--speculation-and-the-vram-levers)
- [Part 5 — Can llama.cpp be adapted?](#part-5--can-llamacpp-be-adapted)
- [What the client blocks today](#what-the-client-blocks-today)
- [Conclusions](#conclusions)
- [Reproducing this](#reproducing-this)

---

## The question

A 24 GB card holds exactly one large model. Qwen3.8-27B-Q4_K_M occupies 21.0 GB
at `ctx_size: 108000` and leaves 2.8 GB — enough for an embedding model and a
0.6B autocompleter, and nothing else. Every agentic turn, including the parts
that are pure bookkeeping, runs on the 27B because there is nothing else to run
them on.

MiniCPM5-2B claims agentic competence at a size that fits in that gap. If the
claim holds, the interesting move is not "a smaller model for small jobs" — it
is splitting one agent across two models so the expensive one only does the part
that needs it.

This study tests that claim end to end: capability first, then co-residency,
then which specific calls in the runtime should move, then whether the VRAM cost
can be engineered away.

## First, a naming correction

MiniCPM**5** and MiniCPM-**V** are unrelated release lines, and conflating them
wastes a day.

| | MiniCPM5 | MiniCPM-V 4.6 |
|---|---|---|
| Modality | **Text only** | Multimodal (image, multi-image, video) |
| Architecture | `LlamaForCausalLM` | SigLIP2-400M + Qwen3.5-0.8B backbone |
| Sizes | 1B (1.08B), 2B (2.52B) | 1.3B |
| Context | 131072 | 262144 |
| Licence | Apache-2.0 | Apache-2.0 |

MiniCPM5-1B shipped first; MiniCPM5-2B followed on 2026-09-07 (2.52B total,
1.98B non-embedding). Nothing in the MiniCPM5 line touches a vision path.

One caveat on the marketing: the "beats Qwen3.5-4B" claim rests on OpenBMB's own
34-benchmark comparison table (average 53.9). This study reproduces the tool-use
behaviour independently. It does not test the general-capability claim.

### The tool-calling caveat that does not apply here

The model card states that tool calling needs SGLang, "which converts the
model's XML-style tool calls to OpenAI-compatible format". That is not true on
llama.cpp build 10434, which carries a dedicated parser:

```
$ strings bin/llama-server | grep -i minicpm
minicpm5
Using specialized template: MiniCPM5
...common_chat_params_init_minicpm5...
```

`tool_calls` arrive already in OpenAI shape, with no extra flags — recent
llama-server defaults to `--jinja`.

## Part 1 — Is the 2B actually agentic?

The reference point is [Qwen3-0.6B](qwen3-0.6b-agentic.md), which handles
single-shot tool calls well and fails on multi-step chains: asked to look up a
user id and *then* an email from that id, it fires both calls in one turn and
invents the intermediate value. That failure is what forces a harness to
single-step dependent calls by hand.

MiniCPM5-2B at `ctx_size: 20000`, f16 KV, through `POST /v1/chat/completions`:

| Test | Result |
|---|---|
| Two independent tool calls in one turn | Both well-formed, valid JSON arguments, empty `content`, `finish_reason: tool_calls` |
| `reasoning_content` separation | Thinking lands in its own field, never inlined into `content` |
| **Sequential dependency** | **One call emitted, then waited** |
| Second leg with the real value | `get_email_by_id{"user_id":84213}` — the actual id, not a guess |
| Third turn synthesis | Correct final answer, `cached_tokens: 336` |

The sequential case is the headline. Given `get_user_id` and `get_email_by_id`
and a system prompt instructing it to wait, it emitted exactly
`get_user_id{"username":"jdoe"}` and stopped. **A naive harness that blindly
executes returned `tool_calls` is safe with this model in a way it is not with
Qwen3-0.6B.**

Cost, RTX 4090, Q4_K_M:

| Metric | Value |
|---|---|
| Prompt throughput | 6,908 tok/s |
| Decode (solo) | 328 tok/s |
| Real VRAM @ ctx 20000 | 2.6 GB |
| Completion for a two-call answer | 345 tokens |

That last row is the thing to plan around: most of those 345 tokens are the
`<think>` trace. **This model reasons before every tool call**, so wall-clock per
agent step is dominated by reasoning, not by the call. It matters again in Part 3.

## Part 2 — Can both models stay resident?

### What the 2B costs

`cache_type_k/v: q8_0`, `n_gpu_layers: 99`, via `scripts/measure_vram.py`:

| ctx_size | real VRAM |
|---|---|
| 8192 | 1.97 GB |
| 16384 | 2.15 GB |
| 32768 | 2.52 GB |
| 65536 | 3.28 GB |
| 131072 | 4.80 GB |

Weights floor ~1.79 GB; q8_0 KV runs ~22 KiB/token, linear. `calibrate` reads
the architecture as `llama • 42L • 2 KV-heads • head_dim 128` — a plain dense
GQA model, so its KV arithmetic is trustworthy here, unlike the hybrid Qwen3.8
architectures where it under-sizes context.

### Naive co-residency OOMs

27B at its production `ctx_size: 108000` (21.0 GB) plus the 2B at 32768
(2.5 GB) **loads** — 24063 MiB used, 501 MiB free — and then dies the moment
both generate at once. A 400-token 27B completion fired concurrently with a 2B
tool call killed the 27B outright:

```
E  cuMemCreate(&handle, reserve_size, &prop, 0)
E  current device: 0, in function alloc at ggml-cuda.cu:589
```

The daemon returned `502 Bad Gateway`; `zallama ps` then showed the 27B with an
em-dash for real VRAM — process gone — while still holding its 21.0 GB budget
reservation. **501 MiB of idle headroom is not enough.** Prior work on the
services group settled on 435 MiB as tight-but-survivable for a 1.4 GB model; a
27B's compute buffers are a different order of magnitude.

### The fix: buy the 2B's VRAM from the 27B's context

`calibrate` on the 27B reports `qwen35 • 65L (17 with KV) • 4 KV-heads •
head_dim 256`, **36 KiB/token** at q8_0. Only 17 of 65 layers carry KV, and it is
still the largest single lever on the card:

| 27B ctx_size | KV cost | vs 108000 |
|---|---|---|
| 108000 | 3.89 GB | — |
| 65536 | 2.36 GB | −1.53 GB |
| 32768 | 1.18 GB | **−2.71 GB** |
| 16384 | 0.59 GB | −3.30 GB |

Dropping the 27B from 108000 to 32768 frees 2.71 GB — almost exactly the 2.52 GB
the 2B costs at the same context. **The two-model split is very nearly
self-financing.** Measured:

| Layout | idle free | peak free under load | 27B decode | 2B decode |
|---|---|---|---|---|
| 27B@108k + 2B@32k | 501 MiB | **OOM — 27B killed** | — | — |
| 27B@32k + 2B@32k | 3703 MiB | **3641 MiB** | 74.3 tok/s | 119.9 tok/s |

Both served concurrently with 3.6 GB to spare.

**Contention is real.** The 2B decodes 328 tok/s alone and 119.9 tok/s while the
27B is generating — they share SMs, not just VRAM. Budget for a ~2.7x slowdown
on the small model whenever the big one is mid-turn, and do not design a UI that
assumes the 2B is instant during a 27B answer.

The architectural justification follows the arithmetic: the 27B's 108k window
exists only because it is currently the sole model in the conversation. Hand the
turn-by-turn conversation to the 2B and the 27B receives a focused reasoning
task, which 32k covers comfortably.

## Part 3 — Which work should move?

### The seam already exists

`zeagentrs` `SessionConfig` carries `router_model: Option<String>`
(`src/agent/mod.rs:299`), and the built-in router takes an explicit model:

```rust
Some(Arc::new(LLMRouter::new(
    config.adapter.clone(),
    config.router_model.clone().unwrap_or_else(|| config.model.clone()),
```

Note `config.adapter.clone()` — the router reuses the **same** adapter. Against
zallama that is exactly right: one `base_url`, different model names, both
resident. A second model costs no adapter work at all.

**zechat never sets `router_model`.** It sets only `router_max_tokens`
(`zechat/src-tauri/src/agent.rs:757`), so routing runs on the 27B today.

### Every LLM call in the runtime

| Site | Function | What it decides | Model today |
|---|---|---|---|
| `execution/router.rs:561` | `route` | mode + categories + skills | `router_model` ?? chat model |
| `agent/mod.rs:1892` | `run_mini_planning_step` | sub-goals for the turn | chat model (fixed) |
| `agent/mod.rs:1994` | `reconcile_sub_goals` | goal progress | chat model (fixed) |
| `agent/mod.rs:2122` | `verify_goal` | is the goal satisfied | chat model (fixed) |
| `agent/mod.rs:3169` | `node_think` | **the answer + tool calls** | chat model (fixed) |
| `memory/mod.rs:480` | `select_for_query` | which memories are relevant | own `model` field |
| `memory/mod.rs:703` | `reflect` | end-of-turn memory extraction | own `model` field |
| `tools/builtin.rs:775` | `SummarizeSandwichTool` | compress bulky tool output | `params.model` ?? `ctx.model` |

Only `node_think` is the product. Everything else is plumbing — which is what
makes the question worth asking.

### Benchmark: the router, 2B vs 27B

Eight cases through the **real** `router.system.md` / `router.user.md` contract
(five tool categories, two skills, a nine-tool always-on floor),
`temperature: 0`. Cases: factual, single-category, multi-category, pronoun
follow-up, memory-driven preference, always-on-tool-only, skill-only, prompt
injection.

| Config | Correct | Wall (8 calls) | Completion tokens | Parse failures |
|---|---|---|---|---|
| **27B, `reasoning: false`** | **8/8** | 6.8 s | 454 | 0 |
| 2B, reasoning on, free-form | 6/8 | 28.1 s | 8089 | 1 |
| 2B, reasoning off, free-form | 5/8 | 1.8 s | 473 | 1 |
| **2B, reasoning off + JSON schema** | 6/8 | **2.0 s** | 508 | **0** |
| 2B, reasoning on + JSON schema | 6/8 | 32.3 s | 8927 | 1 |

Three results fall out of this table.

**Reasoning is the dominant cost, not parameter count.** The 27B is cheap at
routing *because* its registry sets `reasoning: false` — 454 tokens for eight
decisions. The 2B with thinking on spends 8089 and runs 4x slower than a model
thirteen times its size. Any model doing router duty needs reasoning off. (In
the reasoning-on runs, one case blew the 4096-token ceiling entirely and never
emitted JSON — it got stuck ruminating on the prompt-injection case.)

**Most apparent 2B "routing errors" were serialization errors.** Free-form, it
emitted:

```json
{"mode": "task", "categories": "[\"weather\"]", "skills": [], ...}
```

— a JSON *string* containing a JSON array. The decision was correct; the shape
was not. `Vec<String>` will not deserialize from that, so the runtime sees a
routing failure for a right answer. Constrained decoding (`response_format:
json_schema`, with `categories` and `skills` as enums over the live catalogue)
eliminated every shape failure and took 5/8 to 6/8.

**It is still not good enough to take the router.** The two survivors are
precisely the ones the system prompt was written to prevent: the pronoun
follow-up came back with *no categories at all* — the "answering 'no category,
it refers to a previous turn' strips the very tools that turn needs" failure —
and the skill-only turn put `writing` in `categories`. Trading 8/8 for 6/8 to
save 0.6 s per turn is a bad trade, particularly when routing on the 27B is
already only 0.85 s per call.

### The rule: split by judgment density, not token volume

If the agent's control flow branches on the output, it stays on the big model.
If the call compresses or extracts text, it moves.

**Move to the 2B:**

- `SummarizeSandwichTool` — the clearest win. High token volume, no judgment,
  and it *already* accepts a `model` parameter, so this is a caller change, not
  a runtime change.
- `memory::reflect` — end-of-turn, entirely off the critical path. Latency there
  is invisible to the user, and it has its own model slot.
- `memory::select_for_query` — id-picking from a compact index, own model field.

**Keep on the 27B:** `node_think`, `verify_goal`, `reconcile_sub_goals`,
`run_mini_planning_step`, and — on this evidence — `route`.

### Before moving the router, measure it properly

Eight cases is a direction, not a verdict. zechat already has the right
instrument: `tests/agentic-bench.mjs`, which sweeps `settings.json` overlays on a
copy of `~/.zechat` and is explicitly "how a tuning claim gets settled instead of
argued". Two pieces of prerequisite work are worth doing regardless, because they
help the 27B too:

1. Add JSON-schema constrained decoding to the router call, with `categories`
   and `skills` as enums over the live catalogue. This removes a failure class
   rather than trading accuracy for it.
2. Make the deserializer tolerant of a double-encoded array — one
   `string-or-seq` deserializer turns a hard routing failure into a correct
   decision.

## Part 4 — Speculation and the VRAM levers

The natural follow-up is to shrink the 2B's VRAM and buy the lost speed back with
speculative decoding. Four mechanisms were tested: CPU offload, a baked-in MTP
head, the DSpark drafter, and ngram speculation. Two are unavailable on this
model, one is a bad trade at any setting, and one works very well — but only for
the half of the workload this study actually wants to accelerate.

### CPU offload: the worst VRAM per tok/s on the card

`ctx_size: 32768`, q8_0 KV, 42-layer dense model:

| n_gpu_layers | VRAM | decode | vs full GPU |
|---|---|---|---|
| 99 (all) | 2590 MiB | **303.9 tok/s** | — |
| 32 | 2124 MiB | 99.3 tok/s | −466 MiB, **−67%** |
| 24 | 1776 MiB | 67.7 tok/s | −814 MiB, −78% |
| 16 | 1426 MiB | 50.9 tok/s | −1164 MiB, −83% |
| 8 | 1072 MiB | 41.3 tok/s | −1518 MiB, −86% |
| 0 (all CPU) | 502 MiB | 31.4 tok/s | −2088 MiB, −90% |

The cliff is immediate: the *first ten* offloaded layers cost two thirds of the
speed for 18% of the memory.

This is much worse than `n_cpu_moe` on a large MoE, and the reason matters.
`n_cpu_moe` strands only the **expert** weights in system RAM and keeps attention
and the KV cache on the GPU, so per-token GPU work is untouched. Plain layer
offload moves everything in the layer, so every token crosses PCIe. MiniCPM5-2B
is dense — there are no experts to strand, and no cheap version of this knob
exists for it.

### MTP: there is no baked-in head

`grep -ci nextn` over the model's log returns **0** — no ignored `nextn`
tensors — and `calibrate`'s `llama • 42L` is the real layer count, not 41+1.
Unlike Qwen3.8-27B, MiniCPM5-2B-Q4_K_M ships no MTP head, so `spec_type:
draft-mtp` has nothing to activate. See
[MTP](mtp-speculative-decoding.md) for the detection method.

### DSpark: works, but only on structured output

**Corrected 2026-09-11.** An earlier run of this study concluded DSpark was
unusable. That conclusion was wrong, and the cause was operator error: this build
of llama.cpp accepts

```
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,
             ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
```

and the first attempt used `draft-simple`. DSpark is a DFlash-derived
semi-autoregressive drafter, not a plain autoregressive draft model, so
`draft-simple` fed it through the wrong code path and produced 3–7% acceptance.
With `spec_type: draft-dspark` the server says so explicitly:

```
common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dspark'
 - n_max=6, n_min=0, p_min=0.00
 - block_size=7, mask_token_id=75982, n_extract=5
```

Measured on `openbmb/MiniCPM5-2B-DSpark-GGUF` (652 MB), `ctx_size: 8192`:

| spec_type | VRAM | counting | prose | JSON array |
|---|---|---|---|---|
| `none` | 2030 MiB | 303.9 tok/s | 304.0 tok/s | 304.7 tok/s |
| `draft-dspark`, n_max 4 | 3212 MiB | 500.2 (1.65x) | 321.5 (1.06x) | 618.0 (2.03x) |
| `draft-dspark`, **n_max 6** | 3212 MiB | 512.9 (1.69x) | 296.9 (0.98x) | **739.0 (2.42x)** |
| `draft-dspark`, n_max 8 | 3212 MiB | 491.6 (1.62x) | 280.2 (0.92x) | 732.3 (2.40x) |

Draft acceptance tracks the structure of the output, not the length:

| Workload | n_max 4 | n_max 6 | n_max 8 |
|---|---|---|---|
| JSON array | 0.90 | 0.90 | 0.86 |
| counting | 0.73 | 0.61 | 0.55 |
| free prose | 0.31 | 0.23 | 0.20 |

**DSpark is a structured-output accelerator.** At 90% acceptance it gives 2.4x on
JSON and is a slight *loss* on conversational prose, where a longer lookahead
makes it worse. That maps precisely onto the split in Part 3: the 2B's
orchestration work — router decisions, tool calls, memory extraction — is exactly
the JSON-shaped output DSpark doubles, while the conversational turns it would
also be asked to handle get nothing.

The cost is **+1182 MiB** (2030 → 3212), which must be found before the 27B's
context is cut, not after.

### ngram speculation: free, and worth nothing here

The `ngram-*` modes need no draft model and therefore no extra VRAM. On this
model they are flat:

| spec_type | counting | prose |
|---|---|---|
| `none` | 303.9 tok/s | 304.0 tok/s |
| `ngram-simple` | 294.7 | 299.9 |
| `ngram-cache` | 295.8 | 283.1 |
| `ngram-mod` | 302.6 | 303.6 |
| `ngram-map-k` | 302.1 | 302.2 |

Lookup overhead cancels the gain at 300 tok/s on a 2B. Worth re-testing on the
27B, where each verified token is worth ~13x more.

### Correctness: speculation is not output-neutral on a quantized target

[llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618) reports
that `draft-mtp` and `draft-dspark` diverge from vanilla under greedy sampling
when the **target is quantized**, and is open as of this writing. It reproduces
here. Same prompt, `temperature: 0`, `top_k: 1`, `seed: 1`, Q4_K_M target:

| Prompt | Vanilla vs DSpark |
|---|---|
| "Explain in one paragraph how TLS certificate validation works" | **DIVERGED** |
| "List the first 15 prime numbers" | identical |
| JSON router-shaped object | identical |

The prose divergence is not a rounding difference — 106 words versus 74,
sequence similarity 0.578, a different paragraph after a shared opening. Neither
version was wrong; they were simply different generations.

Two consequences:

- The claim in [MTP](mtp-speculative-decoding.md) that "speculative decoding does
  not change *what* the model produces" is **false on this build for a quantized
  target**, and that page should be amended.
- The production `Qwen3.8-27B-Q4_K_M` entry runs `spec_type: draft-mtp` on a
  quantized target — the affected configuration. Nothing observed here suggests
  quality loss, but reproducibility is gone, and any eval comparing that model to
  itself with speculation toggled is not comparing like with like.

The structured workloads stayed identical in every trial, which is consistent
with high acceptance: when the drafter is right 90% of the time there is little
room for the divergent path to be taken.

### The lesson: cut context, never layers

Ranked by VRAM freed per tok/s surrendered:

| Lever | VRAM freed | Speed cost |
|---|---|---|
| **27B `ctx_size` 108000 → 32768** | **2710 MiB** | **none** |
| 2B `ctx_size` 32768 → 8192 | 550 MiB | none |
| 2B `n_gpu_layers` 99 → 32 | 466 MiB | −67% |
| 2B `-ot ffn_down → CPU` (all layers) | 350 MiB | −71% |
| 2B + DSpark draft | **−1182 MiB** (spends) | **+142% on JSON** |

Context is nearly free to cut until the moment the window is genuinely needed;
layers are catastrophic from the first one. The two context levers together free
3.2 GB at zero throughput cost — more than full CPU offload of the 2B
(2088 MiB), while keeping all 303.9 tok/s.

DSpark inverts the sign of the last row: it is the one mechanism here that *buys*
throughput with VRAM rather than the reverse, and on structured output the
exchange rate is good — 1182 MiB for 2.4x. Whether that is affordable is a
question about the 27B's context budget, not about the 2B.

## Part 5 — Can llama.cpp be adapted?

### Shared KV cache between two models: no, and not for a fixable reason

There is a `--kv-unified` flag, and it is not this:

```
-kvu, --kv-unified   use single unified KV buffer shared across all sequences
```

**Sequences, not models** — it unifies the per-slot caches inside one
llama-server instance. Sharing a cache across two *models* is not a missing
feature, it is a category error. A KV entry is a per-layer tensor whose shape is
fixed by that model's architecture, and the two models here agree on nothing:

| | MiniCPM5-2B | Qwen3.8-27B |
|---|---|---|
| Layers | 42 | 65 (17 with KV) |
| KV heads | 2 | 4 |
| head_dim | 128 | 256 |
| KV per token (q8_0) | ~22 KiB | 36 KiB |
| Tokenizer | MiniCPM5 | Qwen |

Different tokenizers alone settle it: the same text is not the same token
sequence, so there is no shared index to cache against. Nothing short of
retraining one model against the other's tokenizer and dimensions makes this
question answerable, at which point they are the same model.

What *is* shareable is the prompt-cache benefit within each model — already
happening, and visible as `cached_tokens: 336` in the Part 1 agent loop.

### Surgical tensor placement: finer knob, same bad trade

`--override-tensor` (`-ot`) places individual tensors by regex, which is the
dense-model analogue of `n_cpu_moe`: keep attention and KV on the GPU, push
selected FFN weights to host RAM. Measured at `ctx_size: 8192`:

| Placement | VRAM | decode |
|---|---|---|
| all on GPU | 2020 MiB | 300.4 tok/s |
| `ffn_down` of layers 30–41 → CPU | 1912 MiB | 176.3 tok/s |
| `ffn_up`+`ffn_gate` of layers 30–41 → CPU | 1856 MiB | 147.5 tok/s |
| all `ffn_down` → CPU | 1670 MiB | 86.4 tok/s |

It is a genuinely finer knob than `n_gpu_layers` — 108 MiB for 41% of the speed
is a granularity `n_gpu_layers` cannot express — but the exchange rate does not
improve. At comparable savings (~17%) it lands where layer offload lands
(−71% versus −67%). **There is no placement that makes host RAM cheap.** The
conclusion from the layer sweep survives: cut context, not weights.

### What llama.cpp has that zallama does not expose

None of the following appear in `LlamaServerBackend._PARAM_MAP` / `_FLAG_MAP`
(`server/backends.py`), so no registry entry can reach them today:

| Flag | What it buys |
|---|---|
| `-ot`, `--override-tensor` | surgical weight placement (above) |
| `-nkvo`, `--no-kv-offload` | KV cache in host RAM — the one lever that attacks the *dominant* term |
| `-kvu`, `--kv-unified` | one KV buffer across slots instead of per-slot |
| `-ctxcp`, `--ctx-checkpoints` | SWA checkpointing |
| `-dev`, `--device` | pin a model to a device |
| `--spec-draft-device`, `--spec-draft-ngl`, `-otd` | place the *draft* model separately from the target |
| `-lm`, `--load-mode` | replaces the deprecated `mlock`/`no_mmap` pair zallama still sets |

Two are worth adding on this evidence. `--spec-draft-ngl` / `--spec-draft-device`
would let the DSpark drafter's 1182 MiB live partly on CPU — the drafter is small
and its errors are cheap, so it is the one component where a bad
speed/VRAM trade is acceptable. And `--no-kv-offload` is the only knob that
touches KV rather than weights, which on the 27B is 3.89 GB at its production
context.

### Unified memory: turning a hard OOM into a soft one

`GGML_CUDA_ENABLE_UNIFIED_MEMORY` and `cudaMallocManaged` are both compiled into
this build's `libggml-cuda.so`. With unified memory enabled, an allocation that
would fail instead migrates pages to host RAM — which is precisely the failure
mode in Part 2, where the 27B died on `cuMemCreate` rather than degrading.

This is not free: UVM page migration under a decode loop can be far slower than
an explicit offload, and it converts a loud failure into a quiet slowdown, which
is arguably worse to operate. But as a **safety net** — a daemon that survives a
mis-sized co-residency instead of returning 502 and leaving a zombie budget
reservation — it is worth measuring. zallama passes no environment to its
backends today, so this needs a small change in `backends.py` before it can even
be tried.

## What the client blocks today

Independent of the runtime work, zechat currently forbids the configuration this
study recommends. `src/modules/chat.js:1822`:

```js
// Local runtimes commonly keep one model loaded at a time. Running
// autocomplete with a different text model forces an unload/reload on
// every pause, which is perceived as unexplained latency.
function alignLocalCompletionModel(...)
  settings.completionModel = chatModel;
```

It force-overwrites `completionModel` to `chatModel` for **any** local profile,
at three enforcement points: boot (`:1839`), model-picker switch (`:2164`), and
the settings select (`:3670`), which rewrites the dropdown value back and toasts
at the user.

**The premise is false for zallama**, which is explicitly multi-model:
`max_loaded_models: 4`, LRU eviction, eviction groups, per-group memory budgets.
The guard defends against a limitation this server does not have, and it is what
blocks the split.

A second gap: the available slots are `chatModel`, `visionModel`,
`completionModel`, `asrModel`, `imageModel`, `embeddingModel`, `ttsModel`. There
is a *completion* slot but no **turn/agent** slot, and `agent.rs:749` hardcodes
`SessionConfig::new(adapter, &settings.chat_model, …)`. "The 2B handles turns,
the 27B handles reasoning" is not a configuration change — it needs a new slot
plus a routing decision in the runtime.

## Conclusions

1. **MiniCPM5-2B is genuinely agentic at 2.6 GB.** It passes the multi-step
   tool-chain test that Qwen3-0.6B fails, at ~330 tok/s solo. That makes it a
   credible default agent brain — resident *beside* a large primary model rather
   than competing with it.
2. **Two models fit, but only if the big one gives up context.** 27B@108k +
   2B@32k OOMs under concurrent load. 27B@32k + 2B@32k runs with 3.6 GB spare.
   The trade is self-financing and architecturally coherent.
3. **Reasoning-on is the real cost driver for orchestration work**, not model
   size. Any model doing structured classification needs thinking disabled.
4. **Constrained decoding is worth adding for its own sake.** It converted most
   of the 2B's apparent routing errors into correct decisions, and would remove
   the same failure class from the 27B.
5. **The router should not move yet** — 8/8 versus 6/8 is not worth 0.6 s.
   Summarization and memory reflection should move now; they are pure
   compression, off the critical path, and already have model slots.
6. **Do not try to engineer the VRAM away — spend it instead.** CPU offload,
   surgical `-ot` placement and ngram speculation all fail to pay. There is no
   baked-in MTP head. But DSpark, driven through the correct `draft-dspark`
   path, buys **2.4x on JSON output for 1182 MiB** — and JSON is exactly what
   the 2B's orchestration role emits. Cutting context remains the only lever
   that frees VRAM at no throughput cost.
7. **Speculation is not output-neutral here.** llama.cpp #25618 reproduces on
   this build: greedy decoding with a quantized target diverges from vanilla on
   free prose, though not on structured output. The production 27B runs
   `draft-mtp` on a Q4_K_M target and is in scope.
8. **Two models cannot share a KV cache**, and `--kv-unified` does not do that —
   it unifies sequences within one model. Different tokenizers and KV geometries
   make the question unanswerable rather than unimplemented.
9. **The client is the current blocker**, not the runtime. `zeagentrs` already
   has the seam; zechat's local-provider guard closes it.

## Reproducing this

```bash
# capability + agentic loop
python3 scripts/agentic_benchmark.py                 # zallama
node tests/agentic-bench.mjs                         # zechat, real binary

# VRAM, per model and per ctx_size
python3 scripts/measure_vram.py <model> [--write]
zallama calibrate <model>                            # arch + KiB/token, no load

# speculation: is there a head to turn on?
LOGS=$(zallama paths | grep -i logs)                  # this install: /bank2/zallama/logs
grep -ci nextn "$LOGS/<model>.log"                    # 0 = no MTP head, nothing to enable
grep -i "draft acceptance" "$LOGS/<model>.log"        # after a speculative run
```

Every number in this report was measured on one RTX 4090 (24564 MiB), zallama
v1.14.0, llama.cpp build 10434 (`7e4c0a968`). The running lab notes, with the
raw session output, are in [minicpm5-agentic.md](minicpm5-agentic.md).
