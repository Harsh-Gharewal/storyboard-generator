"""Pytest configuration — sets environment variables before app imports."""

import os
from unittest.mock import AsyncMock, MagicMock
import pytest_asyncio
from beanie import init_beanie

# Required env vars must be set BEFORE any app module is imported,
# because app.config.Settings() is evaluated at import time.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("DATABASE_NAME", "storyboard_test")
os.environ.setdefault("STORAGE_DIR", "test_storage")

from app.models import ALL_DOCUMENT_MODELS


@pytest_asyncio.fixture(autouse=True)
async def init_beanie_fixture():
    """Initialize Beanie document settings for unit tests without requiring a live MongoDB server."""
    # Create a mock collection and database for Beanie initialization
    mock_collection = MagicMock()
    mock_collection.name = "test_collection"
    mock_collection.index_information = AsyncMock(return_value={})

    mock_db = MagicMock()
    mock_db.name = "storyboard_test"
    mock_db.command = AsyncMock()
    mock_db.list_collection_names = AsyncMock(return_value=[])
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    # Disable driver metadata appending on mock client
    mock_client = MagicMock()
    mock_client.append_metadata = MagicMock()
    mock_db.client = mock_client

    await init_beanie(
        database=mock_db,
        document_models=ALL_DOCUMENT_MODELS,
        skip_indexes=True,
    )
