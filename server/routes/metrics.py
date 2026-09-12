"""
routes/metrics.py — GET /metrics (Prometheus text exposition)

Served on the admin listener only. Unauthenticated, like /health: scrapers
on other hosts shouldn't each need the API key, and the body carries only
model names and counters. Restrict reach with the admin port's bind
address / firewall rather than the key.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from ..dependencies import get_pm, get_registry
from .. import metrics

router = APIRouter()

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def prometheus_metrics(pm=Depends(get_pm), registry=Depends(get_registry)):
    return PlainTextResponse(metrics.render(pm, registry), media_type=CONTENT_TYPE)
