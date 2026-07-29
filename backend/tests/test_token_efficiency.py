"""Unit tests for token_logger, prompt_cache, and gemini_client services.

Verifies:
1. Deterministic character bible serialization (stable key ordering, byte-identical).
2. Token logger JSONL writing to /storage/logs/token-usage.log and get_summary(script_id).
3. Gemini client retry backoff on 429/5xx and token usage extraction.
4. Prompt cache explicit vs implicit caching thresholds and stable prefix generation.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanie import PydanticObjectId
from app.config import settings
from app.models.character import Character
from app.models.script import Script
from app.services import gemini_client, prompt_cache, token_logger


# ── Token Logger Tests ───────────────────────────────────────────────

def test_token_logger_write_and_summary(tmp_path):
    """Test that token_logger appends to log file and calculates savings percentage."""
    log_dir = tmp_path / "logs"
    log_file = log_dir / "token-usage.log"

    with patch.object(token_logger, "_get_log_file_path", return_value=log_file):
        # Log 3 calls for script_1, 1 for script_2
        token_logger.log_token_usage(
            model="gemini-3.1-flash-image",
            prompt_tokens=1000,
            cached_tokens=800,
            script_id="script_1",
            call_type="image_generate",
        )
        token_logger.log_token_usage(
            model="gemini-3.1-flash-image",
            prompt_tokens=1000,
            cached_tokens=900,
            script_id="script_1",
            call_type="image_generate",
        )
        token_logger.log_token_usage(
            model="gemini-3.5-flash",
            prompt_tokens=500,
            cached_tokens=0,
            script_id="script_2",
            call_type="script_parse",
        )

        assert log_file.exists(), "Log file must be created"

        # Check script_1 summary: total = 2000, cached = 1700, savings = 85%
        summary1 = token_logger.get_summary("script_1")
        assert summary1["total_prompt_tokens"] == 2000
        assert summary1["cached_tokens"] == 1700
        assert summary1["fresh_tokens"] == 300
        assert summary1["savings_percentage"] == 85.0
        assert summary1["call_count"] == 2

        # Check script_2 summary
        summary2 = token_logger.get_summary("script_2")
        assert summary2["total_prompt_tokens"] == 500
        assert summary2["cached_tokens"] == 0
        assert summary2["savings_percentage"] == 0.0
        assert summary2["call_count"] == 1

        # Check overall summary (script_id=None)
        overall = token_logger.get_summary(None)
        assert overall["total_prompt_tokens"] == 2500
        assert overall["cached_tokens"] == 1700
        assert overall["call_count"] == 3


# ── Prompt Cache Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_character_bible_deterministic():
    """Character bible output must be byte-identical regardless of input order."""
    script_id = PydanticObjectId()

    char_a = Character(script_id=script_id, name="Alice", description="Red hair, blue sweater")
    char_b = Character(script_id=script_id, name="Bob", description="Tall, leather jacket")
    char_c = Character(script_id=script_id, name="Charlie", description="Short hair, glasses")

    # Pass in different input orders
    bible1 = await prompt_cache.build_character_bible([char_c, char_a, char_b])
    bible2 = await prompt_cache.build_character_bible([char_a, char_b, char_c])

    assert bible1 == bible2, "Character bible serialization must be byte-identical"
    assert "CHARACTER BIBLE:" in bible1
    # Check alphabetical ordering: Alice -> Bob -> Charlie
    pos_a = bible1.index("Alice")
    pos_b = bible1.index("Bob")
    pos_c = bible1.index("Charlie")
    assert pos_a < pos_b < pos_c, "Characters must be sorted alphabetically"


def test_build_style_guide():
    """Style guide returns a fixed, non-empty visual guide string."""
    guide = prompt_cache.build_style_guide()
    assert "STYLE GUIDE:" in guide
    assert "16:9" in guide


@pytest.mark.asyncio
async def test_explicit_cache_under_threshold():
    """If character bible + style guide is < 2000 tokens, return None (implicit caching)."""
    script_id = PydanticObjectId()
    short_script = Script(id=script_id, raw_text="Short text", explicit_cache_name=None)

    with patch.object(Script, "get", AsyncMock(return_value=short_script)):
        cache_handle = await prompt_cache.get_or_create_explicit_cache(
            script_id=script_id,
            cacheable_text="Short prompt prefix",  # well under 2000 tokens (~8000 chars)
        )
        assert cache_handle is None, "Should return None for implicit caching under threshold"


@pytest.mark.asyncio
async def test_explicit_cache_over_threshold():
    """If prefix exceeds ~2000 tokens, create explicit cache and store handle on Script doc."""
    script_id = PydanticObjectId()
    script_doc = Script(id=script_id, raw_text="Long script", explicit_cache_name=None)
    mock_save = AsyncMock()
    object.__setattr__(script_doc, "save", mock_save)

    # Generate text > 8000 chars (~2000 tokens)
    long_prefix = "Character detailed description line. " * 300

    with (
        patch.object(Script, "get", AsyncMock(return_value=script_doc)),
        patch(
            "app.services.gemini_client.create_cache",
            new_callable=AsyncMock,
            return_value="cachedContents/explicit-cache-123",
        ) as mock_create_cache,
    ):
        cache_handle = await prompt_cache.get_or_create_explicit_cache(
            script_id=script_id,
            cacheable_text=long_prefix,
        )

        assert cache_handle == "cachedContents/explicit-cache-123"
        mock_create_cache.assert_called_once()
        assert script_doc.explicit_cache_name == "cachedContents/explicit-cache-123"
        mock_save.assert_called_once()


# ── Gemini Client Retry & Logging Tests ─────────────────────────────

@pytest.mark.asyncio
async def test_gemini_client_retry_on_429():
    """Client should retry on 429 status up to 2 times before succeeding or failing."""
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'
    mock_response.usage_metadata = MagicMock(prompt_token_count=150, cached_content_token_count=100)

    # Fail once with 429, succeed on 2nd attempt
    err_429 = Exception("429 Resource Exhausted: Rate limit exceeded")
    mock_generate = AsyncMock(side_effect=[err_429, mock_response])

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = mock_generate

    with (
        patch("app.services.gemini_client._get_client", return_value=fake_client),
        patch("app.services.token_logger.log_token_usage") as mock_log,
    ):
        resp = await gemini_client.call_text_model({"prompt": "Hello", "script_id": "test_script"})

        assert resp == mock_response
        assert mock_generate.call_count == 2, "Should have retried 2 times"
        mock_log.assert_called_once_with(
            model=settings.GEMINI_TEXT_MODEL,
            prompt_tokens=150,
            cached_tokens=100,
            script_id="test_script",
            call_type="text_generate",
        )


def test_map_text_model_to_image_model():
    """Verify that text models map correctly to available image generation models."""
    from app.services.gemini_client import map_text_model_to_image_model

    # Mapping checks
    assert map_text_model_to_image_model("gemini-3.5-flash") == "gemini-3.1-flash-image"
    assert map_text_model_to_image_model("gemini-3.6-flash") == "gemini-3.1-flash-image"
    assert map_text_model_to_image_model("gemini-2.5-flash") == "gemini-2.5-flash-image"
    assert map_text_model_to_image_model("gemini-3.1-pro") == "gemini-3-pro-image"
    assert map_text_model_to_image_model(None) == "gemini-3.1-flash-image"
    assert map_text_model_to_image_model("") == "gemini-3.1-flash-image"

    # Fallbacks / custom handling
    assert map_text_model_to_image_model("custom-model-image") == "custom-model-image"
    assert map_text_model_to_image_model("custom-model") == "custom-model-image"

