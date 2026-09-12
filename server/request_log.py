"""
Per-request metrics for `zallama monitor`.

Every proxied /v1 request is registered here when it starts and finalised
when it ends, so the monitor can show what is in flight right now and what
the last N requests cost. For chat/completions the numbers come off
llama.cpp's own `timings` block (prompt/decode rates, cache hits, draft
acceptance) plus a client-side TTFT taken when the first content delta goes
by; the other endpoints (embeddings, rerank, audio, images) only have a
duration to report.

Kept in memory only — a bounded deque — because the point is a live view,
not an audit trail. Nothing here is on the hot path in any measurable way:
the streaming parser looks at each SSE line once and gives up on anything
that isn't JSON.

Alongside the deque, `Aggregates` keeps monotonically growing per-model
totals and latency histograms for `/metrics` (Prometheus). Those are the
one thing here that *is* meant to outlive the last 200 requests: a scraper
turns them into rates, so they must only ever go up.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict


@dataclass
class RequestRecord:
    id: int
    model: str
    endpoint: str
    stream: bool
    started_at: float
    finished_at: float | None = None
    status: str = "active"          # active | ok | error
    error: str | None = None
    ttft_ms: float | None = None
    duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    draft_accept: float | None = None
    # Backend-side wall time (ms) for prefill and decode, off llama.cpp's
    # `timings`; summed per model so a scraper can derive average tok/s over
    # any window (rate(tokens)/rate(seconds)) rather than only the last value.
    prefill_ms: float | None = None
    decode_ms: float | None = None
    # Raw speculative-decoding counts behind draft_accept, for the same reason.
    draft_n: int | None = None
    draft_n_accepted: int | None = None
    # Streaming only: tokens seen so far, so an in-flight row can show progress.
    tokens_so_far: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Live duration for rows still in flight.
        if self.finished_at is None:
            d["duration_ms"] = round((time.time() - self.started_at) * 1000)
        return d


# Histogram bucket edges (seconds), Prometheus-style cumulative `le` buckets.
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
DURATION_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)


@dataclass
class Histogram:
    """Cumulative-bucket histogram in the Prometheus sense: `counts[i]` is
    the number of observations <= `edges[i]`; +Inf is implied by `count`."""
    edges: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * len(self.edges)

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for i, edge in enumerate(self.edges):
            if value <= edge:
                self.counts[i] += 1


@dataclass
class ModelAggregate:
    """Everything /metrics reports for one model, all monotonic except `last_*`."""
    requests: dict[tuple[str, str], int] = field(default_factory=dict)  # (endpoint, status) -> n
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    draft_generated: int = 0
    draft_accepted: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    ttft: Histogram = field(default_factory=lambda: Histogram(TTFT_BUCKETS))
    duration: dict[str, Histogram] = field(default_factory=dict)  # endpoint -> hist
    # Most recent values, for the "what is it doing right now" gauges.
    last_prefill_tps: float | None = None
    last_decode_tps: float | None = None
    last_draft_accept: float | None = None


class RequestLog:
    def __init__(self, maxlen: int = 200):
        self._recent: deque[RequestRecord] = deque(maxlen=maxlen)
        self._active: dict[int, RequestRecord] = {}
        self._seq = 0
        self._totals = {"requests": 0, "errors": 0,
                        "prompt_tokens": 0, "completion_tokens": 0}
        self._started = time.time()
        self.aggregates: dict[str, ModelAggregate] = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self, model: str, endpoint: str, stream: bool = False) -> RequestRecord:
        self._seq += 1
        rec = RequestRecord(id=self._seq, model=model, endpoint=endpoint,
                            stream=stream, started_at=time.time())
        self._active[rec.id] = rec
        return rec

    def first_token(self, rec: RequestRecord) -> None:
        if rec.ttft_ms is None:
            rec.ttft_ms = round((time.time() - rec.started_at) * 1000, 1)

    def finish(self, rec: RequestRecord, *, error: str | None = None,
               timings: dict | None = None, usage: dict | None = None) -> None:
        if rec.finished_at is not None:
            return
        rec.finished_at = time.time()
        rec.duration_ms = round((rec.finished_at - rec.started_at) * 1000, 1)
        rec.status = "error" if error else "ok"
        rec.error = (error or None) and str(error)[:200]
        if timings or usage:
            self.apply_metrics(rec, timings or {}, usage or {})
        # A non-streamed call has no first-token moment distinct from the end.
        if rec.ttft_ms is None and not rec.stream and not error:
            rec.ttft_ms = rec.duration_ms
        self._active.pop(rec.id, None)
        self._recent.appendleft(rec)
        self._totals["requests"] += 1
        if error:
            self._totals["errors"] += 1
        self._totals["prompt_tokens"] += rec.prompt_tokens or 0
        self._totals["completion_tokens"] += rec.completion_tokens or 0
        self._aggregate(rec)

    def _aggregate(self, rec: RequestRecord) -> None:
        agg = self.aggregates.get(rec.model)
        if agg is None:
            agg = self.aggregates[rec.model] = ModelAggregate()
        key = (rec.endpoint, rec.status)
        agg.requests[key] = agg.requests.get(key, 0) + 1
        agg.prompt_tokens += rec.prompt_tokens or 0
        agg.completion_tokens += rec.completion_tokens or 0
        agg.cached_tokens += rec.cached_tokens or 0
        agg.prefill_seconds += (rec.prefill_ms or 0) / 1000
        agg.decode_seconds += (rec.decode_ms or 0) / 1000
        if rec.draft_n:
            agg.draft_generated += rec.draft_n
            agg.draft_accepted += rec.draft_n_accepted or 0
        if rec.ttft_ms is not None and rec.stream:
            agg.ttft.observe(rec.ttft_ms / 1000)
        if rec.duration_ms is not None:
            hist = agg.duration.get(rec.endpoint)
            if hist is None:
                hist = agg.duration[rec.endpoint] = Histogram(DURATION_BUCKETS)
            hist.observe(rec.duration_ms / 1000)
        if rec.prefill_tps is not None:
            agg.last_prefill_tps = rec.prefill_tps
        if rec.decode_tps is not None:
            agg.last_decode_tps = rec.decode_tps
        if rec.draft_accept is not None:
            agg.last_draft_accept = rec.draft_accept

    def active_by_model(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self._active.values():
            out[rec.model] = out.get(rec.model, 0) + 1
        return out

    @property
    def started(self) -> float:
        return self._started

    @staticmethod
    def apply_metrics(rec: RequestRecord, timings: dict, usage: dict) -> None:
        """Fill token counts and rates from llama.cpp's response fields."""
        rec.prompt_tokens = timings.get("prompt_n") or usage.get("prompt_tokens") or rec.prompt_tokens
        rec.completion_tokens = (timings.get("predicted_n") or usage.get("completion_tokens")
                                 or rec.completion_tokens)
        if "cache_n" in timings:
            rec.cached_tokens = timings.get("cache_n")
        elif usage.get("prompt_tokens_details"):
            rec.cached_tokens = usage["prompt_tokens_details"].get("cached_tokens")
        if timings.get("prompt_per_second"):
            rec.prefill_tps = round(timings["prompt_per_second"], 1)
        if timings.get("predicted_per_second"):
            rec.decode_tps = round(timings["predicted_per_second"], 1)
        if timings.get("draft_n"):
            rec.draft_n = int(timings["draft_n"])
            rec.draft_n_accepted = int(timings.get("draft_n_accepted", 0))
            rec.draft_accept = round(rec.draft_n_accepted / rec.draft_n, 3)
        if timings.get("prompt_ms") is not None:
            rec.prefill_ms = float(timings["prompt_ms"])
        if timings.get("predicted_ms") is not None:
            rec.decode_ms = float(timings["predicted_ms"])

    # -- streaming parser --------------------------------------------------
    def make_sse_tap(self, rec: RequestRecord):
        """Return a `feed(bytes)` that watches an SSE stream go past.

        Chunks from httpx aren't line-aligned, so a partial trailing line is
        carried to the next call. Each complete `data: {...}` line is decoded
        once: the first content/reasoning delta stamps TTFT, `usage` and
        `timings` (llama.cpp puts both in its final chunk) fill the record.
        """
        buf = b""

        def feed(chunk: bytes) -> None:
            nonlocal buf
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line, buf = buf[:nl].strip(), buf[nl + 1:]
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content") \
                            or delta.get("tool_calls") or choice.get("text"):
                        self.first_token(rec)
                        rec.tokens_so_far += 1
                if obj.get("timings") or obj.get("usage"):
                    self.apply_metrics(rec, obj.get("timings") or {}, obj.get("usage") or {})
        return feed

    # -- read side ---------------------------------------------------------
    def snapshot(self, limit: int = 50) -> dict:
        recent = [r.to_dict() for r in list(self._recent)[:limit]]
        active = sorted((r.to_dict() for r in self._active.values()),
                        key=lambda d: d["started_at"])
        return {
            "active": active,
            "recent": recent,
            "totals": {**self._totals, "since": self._started},
        }
