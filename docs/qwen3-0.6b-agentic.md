# What Qwen3-0.6B Can Actually Do in an Agentic Client

*Testing the smallest Qwen3 for tool-calling and multi-step agent loops, not
just autocompletion — worked through against the registered
`qwen3-0.6b-q8_0` on a 4090.*

`qwen3-0.6b-q8_0` is registered with `ctx_size: 1024` for a narrow job —
single-shot autocompletion of zechat input. This session tested it outside
that box: OpenAI-style `tools`/`tool_calls` through llama-server's built-in
Hermes-format parser (no extra flags needed — recent llama-server defaults to
`--jinja` and reads the GGUF's embedded chat template), at `ctx_size: 8192`,
`temperature: 0`, `reasoning: false` (as registered — no `<think>` block).

- [What works](#what-works)
- [What breaks](#what-breaks)
- [Numbers](#numbers)
- [Takeaway](#takeaway)

---

## What works

- **Parallel tool calls.** A prompt needing two independent tools (weather +
  arithmetic) produced two well-formed `tool_calls` in one response, correct
  function names, valid JSON arguments, empty `content`.
- **Tool-result synthesis.** Feeding both tool results back as `role: tool`
  messages produced a correct, concise natural-language answer combining both
  — and the previous turn's 237 prompt tokens came back `cached_tokens: 237`,
  so the follow-up turn only had to prefill 90 new tokens.
- **Not calling a tool when none is needed.** A plain factual question
  ("capital of Japan?") with the same two tools still available returned
  plain `content`, no spurious `tool_calls` — no false-positive tool use in
  this test.
- **Selecting the right tool among distractors.** Given 6 tools (`read_file`,
  `write_file`, `list_dir`, `run_shell`, `git_diff`, `web_search`) and a
  request to read a specific file, it picked exactly `read_file` with the
  correct path argument — no distractor picked, no extra calls.

## What breaks

- **Sequential dependencies collapse into parallel calls with a hallucinated
  intermediate value.** Asked to look up a user ID from a username and *then*
  look up that user's email using the ID — with an explicit system-prompt
  instruction to call tools one at a time when a later call depends on an
  earlier result — it fired both `get_user_id` and `get_email_by_id` in the
  same turn, inventing `user_id: 1001` for the second call instead of waiting
  for the first tool's actual result. This is the failure mode that matters
  most for an agent loop: a naive harness that blindly executes whatever
  `tool_calls` come back in one turn will run the second call with a made-up
  argument instead of the real one. **A harness driving this model for
  multi-step tool chains needs to detect same-turn dependent calls and force
  single-stepping**, rather than trusting the model to sequence on its own.
- **`ctx_size: 1024` (the registered production setting) is far too small for
  tool-calling.** A real agentic client's system prompt plus tool schemas
  routinely runs several thousand tokens before the user's first message; the
  daemon log from before this test shows a request rejected outright —
  `request (6708 tokens) exceeds the available context size (1024 tokens)`
  — from whatever client had been pointed at it. At `ctx_size: 8192` none of
  the tests here came close to the limit (max prompt seen: 454 tokens with 6
  tool schemas), so 8192 is a reasonable floor for a small tool-using agent;
  `calibrate` reports this GGUF trained to 40960 and there's VRAM headroom to
  go there if a client needs longer tool-result histories.

## Numbers

All at `ctx_size: 8192`, `q8_0` KV cache, `n_gpu_layers: 99`, RTX 4090,
`temperature: 0`:

| Test | Prompt tokens | Completion tokens | Prompt tok/s | Decode tok/s |
|---|---|---|---|---|
| 2 parallel tool calls | 241 | 44 | 14,436 | 437 |
| Synthesis from tool results (237 cached) | 90 new | 35 | 12,680 | 454 |
| No-tool factual answer | — | — | — | — |
| 6-tool distractor selection | 454 (7 cached) | 22 | 38,586 | 435 |

Decode holds steady around **435–455 tok/s** regardless of tool-schema size —
unsurprising at 0.6B, but worth having measured rather than assumed.

## Takeaway

For a narrow, single-shot tool call (pick the one right function, fill in
arguments from the user's message), `qwen3-0.6b-q8_0` is fast and accurate
enough to be genuinely useful in an agentic client — at several hundred
tokens/sec it's cheap enough to call speculatively. It should **not** be
trusted to plan or sequence a multi-step tool chain on its own; a harness
needs to feed back only one dependent call at a time and treat any same-turn
call whose arguments look derived from another pending call's result as
invalid. The model stays registered at `ctx_size: 1024` for its actual job
(zechat autocompletion); an agentic client should register/load it as a
separate entry (or reload with a bumped `ctx_size`) rather than assuming the
1024-token autocomplete config is enough headroom for tool schemas.
