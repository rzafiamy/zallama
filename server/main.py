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

    binary = resolve_binary(cfg)
    logger.info(f"🦙 llama-server binary: {binary}")

    registry_path = Path(__file__).parent.parent / "models" / "registry.yaml"
    registry = ModelRegistry(registry_path, cfg["zallama"]["models_dir"])

    pm = ProcessManager(
        cfg=cfg,
        binary=binary,
        logs_dir=cfg["zallama"]["logs_dir"],
    )

    dependencies.set_pm(pm)
    dependencies.set_registry(registry)

    logger.info(f"✅ Zallama ready — {len(registry.list_models())} model(s) registered")

    # Background idle sweep task
    idle_timeout = cfg["llama_server"].get("idle_timeout", 300)

    async def idle_sweep_loop():
        while True:
            await asyncio.sleep(30)
            await pm.sweep_idle()

    sweep_task = asyncio.create_task(idle_sweep_loop()) if idle_timeout > 0 else None

    yield

    # Shutdown
    logger.info("Shutting down Zallama...")
    if sweep_task:
        sweep_task.cancel()
    await pm.shutdown_all()
    logger.info("Goodbye! 👋")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(cfg: dict) -> FastAPI:
    app = FastAPI(
        title="Zallama",
        description="Ollama-compatible local LLM server powered by llama.cpp",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.cfg = cfg

    # CORS — allow all origins for local use
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
