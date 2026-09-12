"""
server/main.py — Zallama Daemon

FastAPI application entry point:
  - Loads config
  - Initializes ProcessManager and ModelRegistry
  - Registers routes on two listeners:
      inference (zallama.port)      — /v1/*, /health, /
      admin     (zallama.admin_port) — /api/*, /metrics, /health, /docs
  - Starts idle sweep background task

Two ports, one process: the split exists so the inference API can be exposed
to clients while model management and Prometheus scraping stay on a port
that can be firewalled separately. Both apps share the same ProcessManager /
registry singletons (server.dependencies); only the inference app runs the
lifespan that creates them.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Ensure server package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import __version__
from server.config import load_config, resolve_binary
from server.model_registry import ModelRegistry
from server.process_manager import ProcessManager
from server import dependencies
from server.routes import openai as openai_routes
from server.routes import models as model_routes
from server.routes import health as health_routes
from server.routes import zvec as zvec_routes
from server.routes import metrics as metrics_routes


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zallama")


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = app.state.cfg
    log_level = cfg["zallama"].get("log_level", "info").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    try:
        binary = resolve_binary(cfg)
        logger.info(f"🦙 llama-server binary: {binary}")
    except FileNotFoundError as e:
        logger.warning(f"⚠ {e}")

    # The registry lives *beside* the models it references, inside models_dir, so
    # registry + model files are one self-contained unit: they survive a repo
    # re-clone and always stay consistent with the configured models_dir.
    models_dir = Path(cfg["zallama"]["models_dir"])
    registry_path = models_dir / "registry.yaml"
    registry = ModelRegistry(registry_path, str(models_dir))

    pm = ProcessManager(
        cfg=cfg,
        logs_dir=cfg["zallama"]["logs_dir"],
    )

    from server.download_manager import DownloadManager
    dm = DownloadManager(registry, cfg["zallama"]["models_dir"])

    # zvec vector store (RAG). Single SQLite file under rag.zvec_dir; the routes
    # layer embeds/queries through the existing /v1 models.
    from server.zvec.store import init_store
    init_store(cfg["rag"]["zvec_dir"])

    dependencies.set_pm(pm)
    dependencies.set_registry(registry)
    dependencies.set_dm(dm)

    logger.info(f"✅ Zallama ready — {len(registry.list_models())} model(s) registered")

    # Background idle sweep task
    idle_timeout = cfg["llama_server"].get("idle_timeout", 300)

    async def idle_sweep_loop():
        while True:
            await asyncio.sleep(30)
            await pm.sweep_idle()

    sweep_task = asyncio.create_task(idle_sweep_loop()) if idle_timeout > 0 else None

    # Pre-warm pinned models in the background so their (slow, CPU-bound) cold
    # load happens at boot rather than on the first user request. Done off the
    # critical path: the server starts accepting connections immediately and the
    # pinned models become ready shortly after.
    prewarm_task = asyncio.create_task(pm.prewarm_pinned(registry))

    yield

    # Shutdown
    logger.info("Shutting down Zallama...")
    if sweep_task:
        sweep_task.cancel()
    prewarm_task.cancel()
    await pm.shutdown_all()
    logger.info("Goodbye! 👋")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def _install_auth(app: FastAPI, cfg: dict, public_prefixes: tuple[str, ...]) -> None:
    """Optional API-key auth, required everywhere except `public_prefixes`
    and the landing page, so a network-facing deployment leaks nothing (not
    even the API schema) without the key. Preferred form:
    zallama.api_key_sha256 holds only the SHA-256 hex digest of the key (set
    via `zallama apikey`), so the config file never contains the secret
    itself. Plaintext zallama.api_key is still honored; the hashed form wins
    if both are set. Loopback clients are always let through."""
    api_key_sha256 = (cfg["zallama"].get("api_key_sha256") or "").strip().lower()
    api_key_plain = (cfg["zallama"].get("api_key") or "").strip()
    if not (api_key_sha256 or api_key_plain):
        return

    import hashlib
    import secrets
    from datetime import datetime, timezone

    from fastapi import Request
    from fastapi.responses import JSONResponse

    expected_digest = api_key_sha256 or hashlib.sha256(api_key_plain.encode()).hexdigest()

    # Optional expiry instant (ISO-8601 UTC, written by `zallama apikey`).
    # After it passes, the key is rejected until a new one is issued.
    expires_at = None
    expires_raw = (cfg["zallama"].get("api_key_expires") or "").strip()
    if expires_raw:
        expires_at = datetime.strptime(
            expires_raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    loopback_hosts = {"127.0.0.1", "::1"}

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        # CORS preflight requests never carry an Authorization header, so
        # they must pass through untouched or the browser blocks every
        # non-simple cross-origin request (e.g. JSON POST to /v1/*).
        if request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith(public_prefixes):
            return await call_next(request)
        client = request.client
        if client and client.host in loopback_hosts:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        digest = hashlib.sha256(token.encode()).hexdigest()
        if not secrets.compare_digest(digest, expected_digest):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
        if expires_at and datetime.now(timezone.utc) > expires_at:
            return JSONResponse(
                status_code=401,
                content={"detail": "API key expired — issue a new one with `zallama apikey`"})
        return await call_next(request)


def _install_cors(app: FastAPI) -> None:
    # CORS — allow all origins for local UI use. Note: credentials cannot be
    # combined with wildcard origins per the CORS spec, so we leave them off.
    # Added after the API-key middleware so it wraps it: a 401 response then
    # still carries Access-Control-Allow-Origin and the browser surfaces the
    # real status instead of a misleading "CORS error".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _landing(title: str, links: list[tuple[str, str]]) -> str:
    items = "".join(f'<li><a href="{href}" style="color:#58a6ff">{text}</a></li>'
                    for href, text in links)
    return f"""
    <html><body style="font-family:sans-serif;padding:2rem;background:#0d1117;color:#e6edf3">
    <h1>🦙 Zallama</h1>
    <p>{title}</p>
    <ul>{items}</ul>
    </body></html>
    """


def create_app(cfg: dict) -> FastAPI:
    """The inference listener: OpenAI-compatible /v1 API (+ /v1/zvec)."""
    app = FastAPI(
        title="Zallama",
        description="OpenAI-compatible local LLM server powered by llama.cpp",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.cfg = cfg
    _install_auth(app, cfg, public_prefixes=("/health",))
    _install_cors(app)

    app.include_router(health_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(zvec_routes.router)

    admin_url = f"http://{cfg['zallama']['admin_host']}:{cfg['zallama']['admin_port']}"
    if cfg["zallama"]["admin_host"] in ("0.0.0.0", "::"):
        admin_url = f"http://127.0.0.1:{cfg['zallama']['admin_port']}"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return HTMLResponse(content=_landing(
            "OpenAI-compatible API server powered by llama.cpp", [
                ("/docs", "Swagger UI"),
                ("/v1/models", "GET /v1/models"),
                (f"{admin_url}/", f"Admin API &amp; metrics on port {cfg['zallama']['admin_port']}"),
            ]))

    return app


def create_admin_app(cfg: dict) -> FastAPI:
    """The admin listener: model management (/api/*) and Prometheus /metrics.

    No lifespan — the singletons are created by the inference app's lifespan
    and reached through server.dependencies. /metrics is public so scrapers
    don't need the key; everything under /api still does."""
    app = FastAPI(
        title="Zallama admin",
        description="Model management and Prometheus metrics",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.cfg = cfg
    _install_auth(app, cfg, public_prefixes=("/health", "/metrics"))
    _install_cors(app)

    app.include_router(health_routes.router)
    app.include_router(model_routes.router)
    app.include_router(metrics_routes.router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return HTMLResponse(content=_landing(
            "Admin API — model management and monitoring", [
                ("/docs", "Swagger UI"),
                ("/api/health", "GET /api/health"),
                ("/api/ps", "GET /api/ps"),
                ("/api/requests", "GET /api/requests"),
                ("/metrics", "GET /metrics (Prometheus)"),
            ]))

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def _serve_all(servers: list[uvicorn.Server]) -> None:
    """Run several uvicorn servers on one event loop, sharing one signal handler.

    uvicorn's own Server.serve() installs its signal handlers per server; with
    two servers the second install would shadow the first and a SIGTERM would
    stop only one listener, leaving the process alive. So install one handler
    that flags every server, then drive their private _serve() directly."""
    def handle_exit(sig, frame):
        for s in servers:
            s.handle_exit(sig, frame)

    original = {sig: signal.signal(sig, handle_exit) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        # _serve() is uvicorn's serve() minus the per-server signal capture;
        # fall back to serve() on a version without it (one listener would
        # then keep the process alive after SIGTERM — better than not booting).
        await asyncio.gather(*(getattr(s, "_serve", s.serve)() for s in servers))
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def main():
    cfg = load_config()
    z = cfg["zallama"]
    host, port = z["host"], int(z["port"])
    admin_host, admin_port = z["admin_host"], int(z["admin_port"])
    log_level = z.get("log_level", "info")

    inference = uvicorn.Server(uvicorn.Config(
        create_app(cfg), host=host, port=port, log_level=log_level, access_log=True))
    # lifespan="off": the admin app has no lifespan of its own, and uvicorn
    # would otherwise log a warning probing for one.
    admin = uvicorn.Server(uvicorn.Config(
        create_admin_app(cfg), host=admin_host, port=admin_port, log_level=log_level,
        access_log=True, lifespan="off"))

    logger.info(f"🚀 Starting Zallama on http://{host}:{port} "
                f"(admin API + /metrics on http://{admin_host}:{admin_port})")
    # Same loop selection uvicorn.run() would make (uvloop when installed).
    try:
        from uvicorn._compat import asyncio_run
        asyncio_run(_serve_all([inference, admin]),
                    loop_factory=inference.config.get_loop_factory())
    except ImportError:  # older uvicorn: plain asyncio loop
        asyncio.run(_serve_all([inference, admin]))


if __name__ == "__main__":
    main()
