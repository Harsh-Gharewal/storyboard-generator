"""FastAPI application entry point.

Sets up CORS for the Vite dev origin, includes all routers,
initializes the MongoDB/Beanie connection on startup, and
exposes /api/health for liveness checks.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db, close_db
from app.routers import script_router, storyboard_router, shot_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: connect DB on startup, close on shutdown."""
    # ── Startup ──
    logger.info("Starting up — connecting to MongoDB …")
    await init_db()

    # Ensure storage directory exists
    storage = Path(settings.STORAGE_DIR)
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "anchors").mkdir(exist_ok=True)
    (storage / "shots").mkdir(exist_ok=True)

    logger.info("Startup complete")
    yield
    # ── Shutdown ──
    await close_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title="AI Storyboard Generator",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.VITE_DEV_ORIGIN,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving for stored images ────────────────────────────

storage_path = Path(settings.STORAGE_DIR)
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")

# ── Routers ──────────────────────────────────────────────────────────

app.include_router(script_router.router)
app.include_router(storyboard_router.router)
app.include_router(shot_router.router)


# ── Health check ─────────────────────────────────────────────────────

@app.get("/api/health", tags=["health"])
async def health_check():
    """Liveness probe — returns 200 if the server is running."""
    return {"status": "healthy"}

@app.get("/health/db", tags=["health"])
async def db_health_check():
    """MongoDB connection health check."""
    import app.db
    if app.db._client is None:
        return {"status": "disconnected", "error": "Client not initialized"}
    
    try:
        await app.db._client.admin.command("ping")
        return {"status": "connected"}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}
