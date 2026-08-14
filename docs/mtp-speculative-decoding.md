# Doubling Decode Speed with a Baked-In MTP Head

*What Multi-Token Prediction is, how to tell whether your GGUF has one, and the
measured cost of every knob around it — worked through on Qwen3.8-27B.*

Some GGUFs ship an extra prediction head — MTP, "Multi-Token Prediction", stored
as `nextn` tensors — that guesses the next few tokens in one pass. llama.cpp can
run it as a self-contained draft model: no second model to download, no second
process, roughly **2.2x the decode speed**. It is off by default, and llama.cpp
tells you so only in a line you have to be looking for.

- [Does my model have one?](#does-my-model-have-one)
- [Turning it on](#turning-it-on)
- [Tuning the lookahead](#tuning-the-lookahead)
- [A worked example: Qwen3.8-27B on a 4090](#a-worked-example-qwen3827b-on-a-4090)
- [Sizing the context around it](#sizing-the-context-around-it)
- [Checklist](#checklist)

---

## Does my model have one?

Load the model and read the first lines of its log (`zallama logs <model>`). An
unused MTP head announces itself as a wall of ignored tensors:

```
W model has unused tensor blk.64.nextn.eh_proj.weight (size = 55705600 bytes) -- ignoring
W model has unused tensor blk.64.nextn.enorm.weight (size = 20480 bytes) -- ignoring
W model has unused tensor blk.64.nextn.hnorm.weight (size = 20480 bytes) -- ignoring
W model has unused tensor blk.64.nextn.shared_head_norm.weight (size = 20480 bytes) -- ignoring
```

Those `nextn` tensors *are* the draft head, sitting in VRAM budget and doing
nothing. `zallama calibrate <model>` reports the same fact from the GGUF header
without loading anything — the architecture line counts the MTP block:

```
│ Architecture  │ qwen35 • 65L (17 with KV) • 4 KV-heads • head_dim 256 │
```

65 blocks for a 64-layer model: the 65th is the MTP head
(`nextn_predict_layers: 1` in the metadata).

Publishers usually flag it in the filename — unsloth ships `*-MTP-GGUF`
variants — but not always. Qwen3.8-27B's plain `Q4_K_M` has one.

## Turning it on

Two registry params:

```yaml
- name: Qwen3.8-27B-Q4_K_M
  params:
    spec_type: draft-mtp     # activates the baked-in head
    spec_draft_n_max: 3      # tokens drafted per step (llama.cpp's default)
```

```bash
zallama set Qwen3.8-27B-Q4_K_M spec_type=draft-mtp spec_draft_n_max=3
zallama reload Qwen3.8-27B-Q4_K_M
```

The ignored-tensor warnings disappear, and llama.cpp starts reporting its
acceptance rate at the end of each request:

```
draft acceptance = 0.70769 ( 184 accepted / 260 generated), mean len = 3.83
```

Speculative decoding does not change *what* the model produces — rejected drafts
are discarded and the target model's own distribution is preserved. It only
changes how many tokens come out per forward pass.

**Budget about 1 GiB of VRAM for it.** The head's weights load, and the MTP block
gets its own KV cache. Measured on Qwen3.8-27B at `ctx_size: 65536`: 19.2 GiB
off, 20.3 GiB on.

## Tuning the lookahead

`spec_draft_n_max` is the number of tokens drafted per step. Longer drafts win
more when they land and cost more when they don't, and acceptance decays fast.
Measured on Qwen3.8-27B over one fixed three-prompt set at `--temp 0`:

| `spec_draft_n_max` | 2 | **3** | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| decode tok/s | 95.8 | **106.5** | 102.0 | 97.5 | 92.2 | 91.9 |
| draft acceptance | 0.80 | 0.73 | 0.62 | 0.54 | 0.49 | 0.44 |
| VRAM GiB | 20.1 | 20.3 | 20.4 | 20.6 | 20.7 | — |

The default of 3 wins. Try 4 on very predictable output (code, structured
formats), but measure it — see the warning below before you trust a single run.

> **`zallama bench` alone will not settle this.** With the default fixed-length
> generation, each run generates *different* text, and acceptance depends
> entirely on how predictable that text is. The spread swamps the effect:
> `spec_draft_n_max=4` measured 126.2 ±30.7 tok/s in one sweep and 117.0 ±26.8 in
> another, against 92.7 ±8.3 for `n_max=3` — which then beat it on a fixed prompt
> set. Compare candidates on the *same* prompts at `--temp 0`, or raise `--runs`
> until the stdev is small next to the gap you're chasing.

## A worked example: Qwen3.8-27B on a 4090

The registry entry had no `params` at all, so the model ran on the 8192-token
default and truncated conversations (`truncated = 1` in its log).

```yaml
- name: Qwen3.8-27B-Q4_K_M
  file: /bank2/zallama/models/Qwen3.8-27B-Q4_K_M.gguf
  artifacts:
    mmproj: /bank2/zallama/models/mmproj-BF16-Qwen3.8-27B-Q4_K_M.gguf
  mem_gb: 21.0
  params:
    ctx_size: 65536
    cache_type_k: q8_0
    cache_type_v: q8_0
    spec_type: draft-mtp
    spec_draft_n_max: 3
```

| | before | after |
|---|---|---|
| decode, greedy | 47.9 tok/s | **106.5 tok/s** (2.2x) |
| decode, temp 1.0 / top_p 0.95 / top_k 20 | ~48 tok/s | **112.5 tok/s**, 71% acceptance |
| context | 8192, truncating | 65536 |
| VRAM | 17.3 GiB | 20.3 GiB |

Prefill is unchanged at ~2500 tok/s — MTP is a decode-side win only. Vision still
works: an image round-trips through the `mmproj` path with MTP active.

Two knobs that turned out **not** to matter here, both worth skipping on your own
runs:

- **`ubatch_size`** 512 / 1024 / 2048 → 2493 / 2491 / 2455 tok/s prefill, at up
  to +0.8 GiB. Prefill is already compute-bound. Leave it alone.
- **f16 vs q8_0 KV** → 109.3 vs 106.1 tok/s, for +874 MiB. A 3% decode gain is
  not worth halving the context that fits.

## Sizing the context around it

MTP adds one more attention layer to the KV cache, which `zallama calibrate`
accounts for. The thing to watch on **hybrid** architectures — Qwen3.x
interleaves SSM/linear layers with full attention, `full_attention_interval: 4`
— is that only a quarter of the blocks cache anything, so the context that fits
is far larger than a per-block estimate suggests:

```
bytes/token = n_full_attention_layers x n_head_kv x (k_dim x k_bytes + v_dim x v_bytes)
```

For Qwen3.8-27B with a q8_0 cache: `17 x 4 x (256 + 256) x 1.0625` = **36
KiB/token**. Charge all 65 blocks at f16 instead and you get 260 KiB/token —
seven times the real figure, which is how the same 24 GiB card gets sized for
10240 tokens of context when 65536 fits with room to spare.
`zallama calibrate` does this arithmetic for you; see
[Fitting Your Models on One GPU](vram-planning.md) for the rest of the VRAM
picture.

## Checklist

1. `zallama logs <model>` — any `unused tensor blk.N.nextn.*` lines? You have an
   MTP head.
2. `zallama set <model> spec_type=draft-mtp` and reload. Expect ~2x decode and
   ~1 GiB more VRAM.
3. Leave `spec_draft_n_max` at 3 unless a fixed-prompt, `--temp 0` comparison
   says otherwise.
4. Re-run `zallama calibrate <model>` — the extra GiB comes out of the context
   budget.
5. Update `mem_gb` from what `zallama ps` reports, so eviction schedules on the
   real number.
