"""Thin async wrapper around both Gemini models + explicit cache management.

Isolates all direct google-genai SDK calls and exposes retry-wrapped functions:

- call_text_model(payload)   – gemini-3.5-flash for text/JSON tasks.
- call_image_model(payload)  – gemini-3.1-flash-image for image generation/editing.

Both methods log usage_metadata (prompt_tokens vs cached_tokens) to token_logger.
Includes retry with exponential backoff (max 3 attempts) on 429/5xx status codes.
"""

import asyncio
import logging
from typing import Any, Optional, Union

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.services import token_logger

logger = logging.getLogger(__name__)

# ── Client singleton ─────────────────────────────────────────────────

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Return (or create) the singleton genai Client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


# ── OpenAI Mock response types and Client getter ──────────────────────

_openai_client = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        import os
        api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


class OpenAIUsageMetadata:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens
        self.cached_content_token_count = 0


class OpenAITextResponse:
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.text = text
        self.usage_metadata = OpenAIUsageMetadata(prompt_tokens, completion_tokens)


class OpenAIInlineData:
    def __init__(self, image_bytes: bytes):
        self.data = image_bytes


class OpenAIPart:
    def __init__(self, image_bytes: bytes):
        self.inline_data = OpenAIInlineData(image_bytes)
        self.bytes = image_bytes


class OpenAIContent:
    def __init__(self, image_bytes: bytes):
        self.parts = [OpenAIPart(image_bytes)]


class OpenAICandidate:
    def __init__(self, image_bytes: bytes):
        self.content = OpenAIContent(image_bytes)


class OpenAIImageResponse:
    def __init__(self, image_bytes: bytes):
        self.candidates = [OpenAICandidate(image_bytes)]
        self.usage_metadata = OpenAIUsageMetadata(0, 0)


def _is_retryable_error(exc: Exception) -> bool:
    """Predicate function: return True for 429 rate limit or 5xx server errors."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in (429, 500, 502, 503, 504):
        logger.warning("Retrying Gemini call due to status code %s: %s", status_code, exc)
        return True

    msg = str(exc).lower()
    retryable_keywords = [
        "429", "500", "502", "503", "504",
        "resource_exhausted", "quota", "rate limit",
        "service unavailable", "internal server error",
        "overloaded", "deadline_exceeded"
    ]
    if any(kw in msg for kw in retryable_keywords):
        logger.warning("Retrying Gemini call due to error message: %s", exc)
        return True

    return False


# ── Text generation (gemini-3.5-flash) ───────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_retryable_error),
    reraise=True,
)
async def _execute_text_call_with_retry(client, model_name, contents, config_kwargs):
    """Execute the text call with retry logic."""
    if "3.5" in model_name:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    elif "thinking_config" in config_kwargs:
        del config_kwargs["thinking_config"]
        
    return await client.aio.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )

async def call_text_model(
    payload: Union[dict[str, Any], str],
) -> types.GenerateContentResponse:
    """Call gemini text model with retry logic, automatic fallback, and token logging.

    Args:
        payload: Dict or string. If dict, can include:
            - "prompt": str (required if payload is dict)
            - "system_instruction": Optional[str]
            - "response_mime_type": str (default "application/json")
            - "response_schema": Optional[Any]
            - "cached_content_name": Optional[str]
            - "script_id": Optional[str]

    Returns:
        GenerateContentResponse object containing text and usage_metadata.
    """
    if isinstance(payload, str):
        params = {"prompt": payload}
    else:
        params = dict(payload)

    prompt = params.get("prompt", "")
    system_instruction = params.get("system_instruction")
    response_mime_type = params.get("response_mime_type", "application/json")
    response_schema = params.get("response_schema")
    cached_content_name = params.get("cached_content_name")
    script_id = params.get("script_id")

    client = _get_client()

    config_kwargs: dict[str, Any] = {
        "response_mime_type": response_mime_type,
    }

    if system_instruction is not None:
        config_kwargs["system_instruction"] = system_instruction
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema
    if cached_content_name is not None:
        config_kwargs["cached_content"] = cached_content_name

    primary_model = settings.GEMINI_TEXT_MODEL
    if script_id:
        from app.models.script import Script
        from beanie import PydanticObjectId
        try:
            script = await Script.get(PydanticObjectId(str(script_id)))
            if script and script.model:
                primary_model = script.model.lower().strip()
        except Exception as e:
            logger.warning("Could not load Script %s to get model in call_text_model: %s", script_id, e)

    if primary_model.startswith("gpt"):
        # Route to OpenAI Async client
        try:
            openai_client = _get_openai_client()
            openai_text_model = "gpt-4o"
            messages = []
            if system_instruction is not None:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            if response_schema:
                response = await openai_client.beta.chat.completions.parse(
                    model=openai_text_model,
                    messages=messages,
                    response_format=response_schema
                )
                text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            else:
                response = await openai_client.chat.completions.create(
                    model=openai_text_model,
                    messages=messages,
                    response_format={"type": "json_object"} if response_mime_type == "application/json" else None
                )
                text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            openai_resp = OpenAITextResponse(text, prompt_tokens, completion_tokens)

            # Log token usage
            token_logger.log_token_usage(
                model=primary_model,
                prompt_tokens=prompt_tokens,
                cached_tokens=0,
                script_id=script_id,
                call_type="text_generate",
            )
            return openai_resp
        except Exception as e:
            logger.error("OpenAI text call failed: %s", e)
            raise e

    fallback_model = "gemini-2.5-flash"
    
    try:
        response = await _execute_text_call_with_retry(client, primary_model, prompt, config_kwargs)
        used_model = primary_model
    except Exception as e:
        logger.error("Primary model %s failed after retries: %s. Falling back to %s", primary_model, e, fallback_model)
        # Attempt fallback
        response = await _execute_text_call_with_retry(client, fallback_model, prompt, config_kwargs)
        used_model = fallback_model

    # Token accounting
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

    token_logger.log_token_usage(
        model=used_model,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        script_id=script_id,
        call_type="text_generate",
    )

    return response


async def text_generate(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    response_mime_type: str = "application/json",
    response_schema: Optional[Any] = None,
    cached_content_name: Optional[str] = None,
    script_id: Optional[str] = None,
) -> types.GenerateContentResponse:
    """Convenience wrapper delegating to call_text_model."""
    payload = {
        "prompt": prompt,
        "system_instruction": system_instruction,
        "response_mime_type": response_mime_type,
        "response_schema": response_schema,
        "cached_content_name": cached_content_name,
        "script_id": script_id,
    }
    return await call_text_model(payload)


def map_text_model_to_image_model(text_model: Optional[str]) -> str:
    """Map a given text model name (like gemini-3.5-flash) to a valid image generation model.

    If the model string is None or empty, falls back to the configured GEMINI_IMAGE_MODEL.
    """
    if not text_model:
        return settings.GEMINI_IMAGE_MODEL

    mo_clean = text_model.lower().strip()

    # Map specific text models to their matching available image models
    if "gpt-image-1.5" in mo_clean:
        return "gpt-image-1.5"
    elif "gpt-image-1" in mo_clean:
        return "gpt-image-1"
    elif "3.6-flash" in mo_clean or "3.5-flash" in mo_clean or "3.1-flash" in mo_clean:
        return "gemini-3.1-flash-image"
    elif "2.5-flash" in mo_clean:
        return "gemini-2.5-flash-image"
    elif "3.1-pro" in mo_clean or "3-pro" in mo_clean:
        return "gemini-3-pro-image"
    elif "pro" in mo_clean:
        return "gemini-3-pro-image"
    elif not mo_clean.endswith("-image"):
        return f"{mo_clean}-image"
    else:
        return mo_clean


# ── Image generation (gemini-3.1-flash-image) ────────────────────────

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception(_is_retryable_error),
    reraise=True,
)
async def call_image_model(
    payload: Union[dict[str, Any], str],
) -> types.GenerateContentResponse:
    """Call gemini-3.1-flash-image to generate or edit an image.

    Args:
        payload: Dict or string. If dict, can include:
            - "prompt": str (required)
            - "reference_images": Optional[list[bytes]] (character anchor image bytes)
            - "cached_content_name": Optional[str] (explicit cache handle)
            - "script_id": Optional[str]
            - "shot_id": Optional[str]

    Returns:
        GenerateContentResponse containing generated image bytes and usage_metadata.
    """
    if isinstance(payload, str):
        params = {"prompt": payload}
    else:
        params = dict(payload)

    prompt = params.get("prompt", "")
    reference_images = params.get("reference_images")
    cached_content_name = params.get("cached_content_name")
    script_id = params.get("script_id")
    shot_id = params.get("shot_id")

    client = _get_client()

    contents: list[Any] = []
    if reference_images:
        for img_bytes in reference_images:
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
    contents.append(prompt)

    config_kwargs: dict[str, Any] = {
        "response_modalities": ["IMAGE"],
    }
    if cached_content_name is not None:
        config_kwargs["cached_content"] = cached_content_name

    primary_image_model = settings.GEMINI_IMAGE_MODEL
    if script_id:
        from app.models.script import Script
        from beanie import PydanticObjectId
        try:
            script = await Script.get(PydanticObjectId(str(script_id)))
            if script:
                primary_image_model = map_text_model_to_image_model(script.model)
        except Exception as e:
            logger.warning("Could not load Script %s to get model in call_image_model: %s", script_id, e)

    if primary_image_model.startswith("gpt"):
        # OpenAI Image Generation route
        import base64
        openai_client = _get_openai_client()
        # Use the actual model name selected by the user (gpt-image-1 or gpt-image-1.5)
        openai_model = primary_image_model

        max_chars = 950 if "gpt-image-1" in openai_model and "1.5" not in openai_model else 3950
        truncated_prompt = prompt[:max_chars]

        # Omit response_format for gpt-image models
        kwargs = {
            "model": openai_model,
            "prompt": truncated_prompt,
            "n": 1,
            "size": "1024x1024",
        }
        if not openai_model.startswith("gpt-image"):
            kwargs["response_format"] = "b64_json"

        try:
            response = await openai_client.images.generate(**kwargs)
            first_img = response.data[0]
            if getattr(first_img, "b64_json", None):
                b64_data = first_img.b64_json
                image_bytes = base64.b64decode(b64_data)
            elif getattr(first_img, "url", None):
                import urllib.request
                with urllib.request.urlopen(first_img.url) as url_resp:
                    image_bytes = url_resp.read()
            else:
                raise ValueError("Neither b64_json nor url present in OpenAI image response")

            openai_resp = OpenAIImageResponse(image_bytes)

            token_logger.log_token_usage(
                model=primary_image_model,
                prompt_tokens=0,
                cached_tokens=0,
                script_id=script_id,
                call_type="image_generate",
            )
            return openai_resp
        except Exception as e:
            logger.error("OpenAI image generation failed: %s", e)
            raise e

    response = await client.aio.models.generate_content(
        model=primary_image_model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
    )

    # Token accounting
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

    token_logger.log_token_usage(
        model=primary_image_model,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        script_id=script_id,
        shot_id=shot_id,
        call_type="image_generate",
    )

    return response


async def image_generate(
    prompt: str,
    *,
    reference_images: Optional[list[bytes]] = None,
    cached_content_name: Optional[str] = None,
    script_id: Optional[str] = None,
    shot_id: Optional[str] = None,
) -> types.GenerateContentResponse:
    """Convenience wrapper delegating to call_image_model."""
    payload = {
        "prompt": prompt,
        "reference_images": reference_images,
        "cached_content_name": cached_content_name,
        "script_id": script_id,
        "shot_id": shot_id,
    }
    return await call_image_model(payload)


# ── Explicit context cache management ────────────────────────────────

async def create_cache(
    *,
    model: str,
    contents: list[Any],
    display_name: str,
    ttl_seconds: int = settings.CACHE_TTL_SECONDS,
) -> str:
    """Create a Gemini explicit context cache and return its resource name."""
    client = _get_client()

    cache = await client.aio.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            contents=contents,
            ttl=f"{ttl_seconds}s",
            display_name=display_name,
        ),
    )
    logger.info("Created explicit Gemini cache '%s' (TTL=%ds)", cache.name, ttl_seconds)
    return cache.name


async def get_or_refresh_cache(cache_name: str) -> Optional[str]:
    """Return cache_name if still valid, or None if expired."""
    client = _get_client()
    try:
        cache = await client.aio.caches.get(name=cache_name)
        if cache:
            return cache.name
    except Exception as e:
        logger.warning("Explicit cache '%s' is no longer valid: %s", cache_name, e)
    return None


async def delete_cache(cache_name: str) -> None:
    """Delete an explicit context cache by name."""
    client = _get_client()
    try:
        await client.aio.caches.delete(name=cache_name)
        logger.info("Deleted explicit Gemini cache '%s'", cache_name)
    except Exception as e:
        logger.warning("Failed to delete cache '%s': %s", cache_name, e)
