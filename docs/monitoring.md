# Monitoring: `/metrics` for Prometheus

The daemon exposes everything `zallama monitor` shows — and the history that
tool can't keep — as a Prometheus text endpoint on the **admin port**:

```
http://<host>:<admin_port>/metrics        # default admin_port = port + 1 → 11436
```

It is unauthenticated (like `/health`) so scrapers on other machines don't
need the API key. The body contains model names, counters and GPU numbers,
nothing else. Restrict who can reach it with the admin port's bind address
(`zallama.admin_host`) or a firewall rule, not with the key.

## Scrape config

```yaml
scrape_configs:
  - job_name: zallama
    scrape_interval: 15s
    static_configs:
      - targets: ["makix:6768"]
        labels: { gpu_host: makix }
```

Rendering is cheap (one `nvidia-smi` call and a walk over in-memory
structs), so 5–15 s intervals are fine.

## What's exposed

Base units throughout: bytes, seconds, ratios 0–1. `model` is the registry
name. Anything named `_declared_` is the daemon's launch-time estimate
(`mem_gb`), everything else is measured.

### Daemon / capacity

| Metric | Type | Meaning |
|---|---|---|
| `zallama_info{version}` | gauge | always 1 |
| `zallama_uptime_seconds` | gauge | since daemon start |
| `zallama_models_registered` | gauge | registry size |
| `zallama_models_loaded` | gauge | running backend processes |
| `zallama_mem_budget_bytes` | gauge | `llama_server.mem_budget_gb` (0 = unlimited) |
| `zallama_mem_declared_bytes` | gauge | sum of `mem_gb` over loaded models — what the eviction logic believes |
| `zallama_vram_used_bytes` | gauge | measured VRAM held by zallama's own backend processes |
| `zallama_group_mem_budget_bytes{group}` | gauge | per-`evict_group` budget |

### Registry — every model, loaded or not

| Metric | Type | Meaning |
|---|---|---|
| `zallama_model_info{model,modality,backend,group,pinned,mem_gb,file,aliases}` | gauge | 1 per registered model |
| `zallama_model_param{model,param,value}` | gauge | 1 per **effective launch parameter** — registry `params` layered over `llama_server.default_params` for the llama-server family (what `zallama show` prints), the registry's own params for other backends; `artifacts` appear as `param="artifact_<kind>"` |
| `zallama_model_param_value{model,param}` | gauge | the numeric ones (`ctx_size`, `n_gpu_layers`, `spec_draft_n_max`, `threads`, `steps`…) as plottable values |
| `zallama_model_file_bytes{model}` | gauge | main model file size on disk |

`zallama_model_param` is the "what was it configured as when that happened"
series: a re-tune (`zallama set … ctx_size=24576`) shows up as the old
label set going stale and a new one appearing, exactly at the timestamp the
tok/s or VRAM graph changed. Join it onto anything per-model with
`group_left`:

```promql
zallama_last_decode_tokens_per_second
  * on (model) group_left (value)
    zallama_model_param{param="spec_type"}
```

### Per loaded instance

| Metric | Type | Meaning |
|---|---|---|
| `zallama_model_loaded{model,modality,backend,group,port}` | gauge | 1 per running backend |
| `zallama_model_mem_declared_bytes{model}` | gauge | the `mem_gb` the daemon budgets for it |
| `zallama_model_vram_bytes{model}` | gauge | measured VRAM of its process (absent for CPU backends such as kokoro) |
| `zallama_model_uptime_seconds{model}` | gauge | since it became healthy |
| `zallama_model_idle_seconds{model}` | gauge | since its last request |
| `zallama_model_active_requests{model}` | gauge | requests currently reading from it |

### Lifecycle counters (survive the instance)

| Metric | Meaning |
|---|---|
| `zallama_model_start_total{model}` | became healthy |
| `zallama_model_start_failure_total{model}` | died or timed out before `/health` — the `503 died during startup` case |
| `zallama_model_crash_total{model}` | found dead on a later request (restarted on the spot) |
| `zallama_model_evict_capacity_total{model}` | LRU-evicted to admit another model |
| `zallama_model_evict_idle_total{model}` | unloaded by the idle sweep |
| `zallama_model_unload_total{model}` | explicit unload via API/CLI |

### Requests (per model)

| Metric | Type | Meaning |
|---|---|---|
| `zallama_requests_total{model,endpoint,status}` | counter | `status` = `ok` / `error` |
| `zallama_requests_active{model}` | gauge | in flight through the proxy |
| `zallama_prompt_tokens_total{model}` | counter | prompt tokens |
| `zallama_completion_tokens_total{model}` | counter | generated tokens |
| `zallama_cached_tokens_total{model}` | counter | prompt tokens served from the KV cache |
| `zallama_prefill_seconds_total{model}` | counter | backend prefill wall time (llama.cpp `prompt_ms`) |
| `zallama_decode_seconds_total{model}` | counter | backend decode wall time (`predicted_ms`) |
| `zallama_draft_tokens_total{model}` | counter | speculative tokens drafted |
| `zallama_draft_accepted_tokens_total{model}` | counter | of which accepted |
| `zallama_last_prefill_tokens_per_second{model}` | gauge | most recent request |
| `zallama_last_decode_tokens_per_second{model}` | gauge | most recent request |
| `zallama_last_draft_acceptance_ratio{model}` | gauge | most recent request |
| `zallama_request_ttft_seconds{model}` | histogram | time to first content delta, streamed requests only |
| `zallama_request_duration_seconds{model,endpoint}` | histogram | end-to-end as seen by the proxy |

Only `/v1` endpoints that go through the proxy are counted: chat/completions,
completions, embeddings, rerank, audio, images. Token and timing fields are
filled for the llama-server family (they come off llama.cpp's `timings`
block); embeddings / audio / images contribute only counts and duration.

### GPU (whole card, `nvidia-smi`)

`zallama_gpu_memory_used_bytes`, `zallama_gpu_memory_total_bytes`,
`zallama_gpu_utilization_ratio`, `zallama_gpu_temperature_celsius`,
`zallama_gpu_power_watts`, all labelled `{gpu, name}`. Used memory is
*all* processes, not just zallama's — it's the number that decides whether
the next model loads. If a `dcgm-exporter` / `node_exporter` already scrapes
the card, ignore these; they exist so a zallama-only setup still sees VRAM.

## Queries worth having

Free VRAM — the value every 503 this month came down to:

```promql
zallama_gpu_memory_total_bytes - zallama_gpu_memory_used_bytes
```

Restart loop (the evict/respawn ping-pong, or a model that no longer fits):

```promql
increase(zallama_model_start_failure_total[5m]) > 2
```

Average decode tok/s over the last 5 minutes, per model (the honest number,
not the last request):

```promql
rate(zallama_completion_tokens_total[5m]) / rate(zallama_decode_seconds_total[5m])
```

Draft acceptance over a window:

```promql
rate(zallama_draft_accepted_tokens_total[15m]) / rate(zallama_draft_tokens_total[15m])
```

p95 TTFT:

```promql
histogram_quantile(0.95, sum by (model, le) (rate(zallama_request_ttft_seconds_bucket[10m])))
```

Which models run with speculative decoding, and at what context:

```promql
zallama_model_param{param="spec_type"}
zallama_model_param_value{param="ctx_size"}
```

How far the daemon's accounting is from reality (positive = `mem_gb` is
under-declared and the model will one day fail to load next to the services):

```promql
zallama_model_vram_bytes - zallama_model_mem_declared_bytes
```

Alert suggestions: free VRAM < 1 GiB for 5 min; `start_failure` increase > 2
in 5 min; `crash` increase > 0; `zallama_models_loaded == 0` while
`zallama_requests_active > 0` for more than the startup timeout.

## Implementation notes

- `server/metrics.py` renders the text format by hand — gauges, counters and
  two histograms from numbers that already exist in `ProcessManager` and
  `RequestLog`. No `prometheus_client`: a registry would just be a second
  copy of that state to keep in sync.
- Lifecycle counters live in `ProcessManager.lifecycle` (a
  `dict[event, Counter[model]]`) and deliberately outlive the instances they
  describe. Request aggregates live in `RequestLog.aggregates`, next to the
  200-entry deque `zallama monitor` reads; the deque is a window, the
  aggregates are monotonic.
- Everything resets when the daemon restarts, which is what Prometheus
  expects from counters (`rate()`/`increase()` handle the reset).
- Histogram buckets: TTFT 50 ms … 30 s, duration 100 ms … 600 s. Change
  `TTFT_BUCKETS` / `DURATION_BUCKETS` in `server/request_log.py` if your
  workload is far from that.
