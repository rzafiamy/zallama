"""
server/main.py — Zallama Daemon

FastAPI application entry point:
  - Loads config
  - Initializes ProcessManager and ModelRegistry
  - Registers routes
  - Starts idle sweep background task
  - Serves embedded Web UI
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
from fastapi.staticfiles import StaticFiles

# Ensure server package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.config import load_config, resolve_binary
from server.model_registry import ModelRegistry
from server.process_manager import ProcessManager
from server import dependencies
from server.routes import openai as openai_routes
from server.routes import models as model_routes
from server.routes import health as health_routes


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
def create_app(cfg: dict) -> FastAPI:
    app = FastAPI(
        title="Zallama",
        description="OpenAI-compatible local LLM server powered by llama.cpp",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.cfg = cfg

    # CORS — allow all origins for local UI use. Note: credentials cannot be
    # combined with wildcard origins per the CORS spec, so we leave them off.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional API-key auth. When zallama.api_key is set, require it as a Bearer
    # token on the proxy/management surfaces. Public/local paths stay open so the
    # bundled Web UI and health checks keep working.
    api_key = (cfg["zallama"].get("api_key") or "").strip()
    if api_key:
        from fastapi import Request
        from fastapi.responses import JSONResponse

        public_prefixes = ("/health", "/ui", "/docs", "/redoc", "/openapi.json")

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            path = request.url.path
            if path == "/" or path.startswith(public_prefixes):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if token != api_key:
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
            return await call_next(request)

    # Routes
    app.include_router(health_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(model_routes.router)

    # Embedded Web UI
    webui_dir = Path(__file__).parent.parent / "webui"
    if webui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(webui_dir), html=True), name="webui")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        webui_index = webui_dir / "index.html"
        if webui_index.exists():
            return webui_index.read_text()
        return HTMLResponse(content="""
        <html><body style="font-family:sans-serif;padding:2rem;background:#0d1117;color:#e6edf3">
        <h1>🦙 Zallama</h1>
        <p>OpenAI-compatible API server powered by llama.cpp</p>
        <ul>
          <li><a href="/docs" style="color:#58a6ff">Swagger UI</a></li>
          <li><a href="/v1/models" style="color:#58a6ff">GET /v1/models</a></li>
          <li><a href="/api/health" style="color:#58a6ff">GET /api/health</a></li>
          <li><a href="/api/ps" style="color:#58a6ff">GET /api/ps</a></li>
        </ul>
        </body></html>
        """)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    host = cfg["zallama"]["host"]
    port = int(cfg["zallama"]["port"])
    log_level = cfg["zallama"].get("log_level", "info")

    app = create_app(cfg)

    logger.info(f"🚀 Starting Zallama on http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
