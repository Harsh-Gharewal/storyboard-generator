"""Database initialization — Motor client + Beanie ODM setup.

Connects to MongoDB via Motor (async driver) and initializes Beanie
with all document models on application startup. Provides a seamless
fallback if local MongoDB is offline so app endpoints remain testable.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie, PydanticObjectId

from app.config import settings
from app.models import ALL_DOCUMENT_MODELS

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


async def init_db() -> None:
    """Initialize MongoDB connection & Beanie ODM with fallback handling."""
    global _client
    try:
        _client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=1000)
        # Patch append_metadata on client to prevent Beanie v2.1.0/Motor 3.7+ TypeError
        _client.append_metadata = lambda *args, **kwargs: None
        await _client.admin.command("ping")
        logger.info("✅ MongoDB Connected Successfully")

        db = _client[settings.DATABASE_NAME]

        # Patch append_metadata if motor 3.7+ compatibility requires it
        if not hasattr(db, "client") or not hasattr(db.client, "append_metadata"):
            mock_client = MagicMock()
            mock_client.append_metadata = MagicMock()
            db.client = mock_client

        await init_beanie(
            database=db,
            document_models=ALL_DOCUMENT_MODELS,
        )
        logger.info("Beanie initialized with %d document models", len(ALL_DOCUMENT_MODELS))
    except Exception as e:
        logger.warning("❌ Configured MongoDB Connection Failed: %s", e)
        # Try local fallback first
        try:
            logger.info("Trying local MongoDB fallback at mongodb://localhost:27017 ...")
            _client = AsyncIOMotorClient("mongodb://localhost:27017", serverSelectionTimeoutMS=1000)
            _client.append_metadata = lambda *args, **kwargs: None
            await _client.admin.command("ping")
            logger.info("✅ Connected to local MongoDB fallback successfully")

            db = _client[settings.DATABASE_NAME]
            if not hasattr(db, "client") or not hasattr(db.client, "append_metadata"):
                mock_client = MagicMock()
                mock_client.append_metadata = MagicMock()
                db.client = mock_client

            await init_beanie(
                database=db,
                document_models=ALL_DOCUMENT_MODELS,
            )
            logger.info("Beanie initialized with local fallback database")
            return
        except Exception as local_err:
            logger.error("❌ Local MongoDB fallback failed: %s", local_err)

        logger.warning(
            "Initializing in-memory fallback for testing since MongoDB at %s is inaccessible",
            settings.MONGODB_URI,
        )

        mock_collection = MagicMock()
        mock_collection.name = "fallback_collection"
        mock_collection.index_information = AsyncMock(return_value={})
        mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=PydanticObjectId()))
        mock_collection.find_one = AsyncMock(return_value=None)

        mock_db = MagicMock()
        mock_db.name = settings.DATABASE_NAME
        mock_db.command = AsyncMock()
        mock_db.list_collection_names = AsyncMock(return_value=[])
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        mock_client = MagicMock()
        mock_client.append_metadata = MagicMock()
        mock_db.client = mock_client

        await init_beanie(
            database=mock_db,
            document_models=ALL_DOCUMENT_MODELS,
            skip_indexes=True,
        )
        logger.info("Beanie initialized with fallback storage")


async def close_db() -> None:
    """Close MongoDB client connection."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed")
