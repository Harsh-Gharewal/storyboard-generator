"""Prompt cache service — builds and manages the stable cached prefix.

Single source of truth for all prompt construction across the application.

Token Efficiency Guarantees:
1. build_character_bible(): Deterministically serializes characters sorted by name
   with stable formatting and zero dynamic values (no IDs, no timestamps) so implicit
   caching matches byte-identical prefixes.
2. build_style_guide(): Fixed, reusable visual style specification.
3. get_or_create_explicit_cache(): If character bible + style guide exceeds ~2,000 tokens,
   uses Gemini's explicit caching API with a 1-hour TTL, saving the handle on the Script doc.
   If under threshold, skips explicit cache creation and relies on implicit caching by
   maintaining prefix position and byte-identity.
4. build_stable_prefix(): Single source of truth function for building prompt prefixes.
"""

import logging
from typing import Any, List, Optional, Union

from beanie import PydanticObjectId

from app.config import settings
from app.models.character import Character
from app.models.scene import Scene
from app.models.script import Script
from app.services import gemini_client

logger = logging.getLogger(__name__)

# ── System Instructions & Style Guide ────────────────────────────────

SYSTEM_INSTRUCTIONS = (
    "SYSTEM INSTRUCTIONS:\n"
    "You are a professional storyboard artist. Generate a single visual storyboard frame "
    "matching the shot description. Maintain exact character visual appearance and costume "
    "as defined in the reference images and character bible. Do not include speech bubbles, "
    "text overlays, or letterboxing."
)


def build_style_guide() -> str:
    """Return the fixed visual style guide for all storyboard generation calls.

    Returns a byte-identical string defining art style, aspect ratio, lighting,
    and rendering constraints.
    """
    return (
        "STYLE GUIDE:\n"
        "- Art Style: Cinematic storyboard illustration frame, semi-realistic rendering\n"
        "- Aspect Ratio: 16:9 widescreen\n"
        "- Framing & Lighting: Medium linework, naturalistic lighting, consistent character scale\n"
        "- Rendering: Grounded natural color palette, clear subject framing, high continuity"
    )


# ── Character Bible Serialization ────────────────────────────────────

async def build_character_bible(
    characters_or_script_id: Union[List[Character], PydanticObjectId, str],
) -> str:
    """Deterministically serialize character details into a stable text block.

    Sorts characters by name alphabetically to guarantee identical byte output
    on every call for a script. Omits Mongo IDs, timestamps, or dynamic fields.

    Args:
        characters_or_script_id: List of Character documents OR script ID to query.

    Returns:
        Formatted character bible string.
    """
    if isinstance(characters_or_script_id, (PydanticObjectId, str)):
        try:
            obj_id = PydanticObjectId(str(characters_or_script_id))
            characters = await Character.find(Character.script_id == obj_id).to_list()
        except Exception as e:
            logger.warning("Failed to fetch characters for script %s: %s", characters_or_script_id, e)
            characters = []
    else:
        characters = list(characters_or_script_id)

    if not characters:
        return "CHARACTER BIBLE:\nNo characters specified."

    # Sort characters deterministically by lowercased name
    sorted_chars = sorted(characters, key=lambda c: c.name.lower())

    lines = ["CHARACTER BIBLE:"]
    for char in sorted_chars:
        desc = char.description.strip() if char.description else "No description provided."
        lines.append(f"- Name: {char.name}\n  Description: {desc}")

    return "\n".join(lines)


# ── Prefix Construction (Single Source of Truth) ─────────────────────

async def build_stable_prefix(
    script_id: PydanticObjectId,
    scene_id: Optional[PydanticObjectId] = None,
) -> str:
    """Build the full stable prompt prefix for a shot generation request.

    Combines:
    1. SYSTEM_INSTRUCTIONS
    2. build_style_guide()
    3. build_character_bible(script_id)
    4. Scene-state block (location, time_of_day, weather, mood) if scene_id given.

    The output is byte-identical across all shots in the same scene, enabling
    implicit prompt caching.

    Args:
        script_id: MongoDB ID of the script.
        scene_id: Optional MongoDB ID of the scene.

    Returns:
        Deterministic prefix string.
    """
    system_part = SYSTEM_INSTRUCTIONS
    style_part = build_style_guide()
    bible_part = await build_character_bible(script_id)

    prefix_parts = [system_part, style_part, bible_part]

    if scene_id is not None:
        try:
            scene = await Scene.get(scene_id)
            if scene:
                scene_lines = [
                    "SCENE CONTEXT:",
                    f"- Location: {scene.location}",
                    f"- Time of Day: {scene.time_of_day or 'unspecified'}",
                    f"- Weather: {scene.weather or 'unspecified'}",
                    f"- Mood: {scene.mood or 'unspecified'}",
                ]
                prefix_parts.append("\n".join(scene_lines))
        except Exception as e:
            logger.warning("Could not append scene context for scene %s: %s", scene_id, e)

    return "\n\n".join(prefix_parts)


# ── Explicit Context Caching Lifecycle ───────────────────────────────

async def get_or_create_explicit_cache(
    script_id: PydanticObjectId,
    cacheable_text: Optional[str] = None,
) -> Optional[str]:
    """Check prefix token size and manage Gemini explicit context cache.

    If cacheable_text (character bible + style guide) exceeds ~2,000 tokens
    (~8,000 characters), creates a Gemini explicit cache with a 1-hour TTL,
    saves the cache handle to script.explicit_cache_name, and returns it.

    If under the ~2,000 token threshold, returns None so callers use implicit
    caching.

    Args:
        script_id: MongoDB ID of the script.
        cacheable_text: Optional text to evaluate/cache. Constructed if None.

    Returns:
        Gemini cache handle string if explicit cache was created/reused, or None.
    """
    script = await Script.get(script_id)
    if not script:
        return None

    # Check if script already has a valid explicit cache name
    if script.explicit_cache_name:
        valid_name = await gemini_client.get_or_refresh_cache(script.explicit_cache_name)
        if valid_name:
            return valid_name
        else:
            logger.info("Existing cache '%s' expired; will recreate", script.explicit_cache_name)
            script.explicit_cache_name = None
            await script.save()

    # Build cacheable text if not provided
    if cacheable_text is None:
        cacheable_text = await build_stable_prefix(script_id)

    # Estimate token count (~4 characters per token heuristic)
    estimated_tokens = len(cacheable_text) // 4

    # Threshold: 2000 tokens (~8000 chars)
    TOKEN_THRESHOLD = 2000
    if estimated_tokens < TOKEN_THRESHOLD:
        logger.info(
            "Cacheable text is ~%d tokens (<%d threshold); using implicit prompt caching",
            estimated_tokens,
            TOKEN_THRESHOLD,
        )
        return None

    # Exceeds threshold -> create explicit cache via Gemini API
    try:
        target_model = gemini_client.map_text_model_to_image_model(script.model)
        cache_name = await gemini_client.create_cache(
            model=target_model,
            contents=[cacheable_text],
            display_name=f"script_cache_{script_id}",
            ttl_seconds=settings.CACHE_TTL_SECONDS,
        )

        script.explicit_cache_name = cache_name
        await script.save()
        logger.info("Explicit cache created for script %s: %s", script_id, cache_name)
        return cache_name
    except Exception as e:
        logger.error("Failed to create explicit Gemini cache for script %s: %s", script_id, e)
        return None


async def ensure_explicit_cache(script_id: PydanticObjectId) -> Optional[str]:
    """Alias for get_or_create_explicit_cache for backwards compatibility."""
    return await get_or_create_explicit_cache(script_id)


async def teardown_cache(script_id: PydanticObjectId) -> None:
    """Delete the explicit context cache for a script and clear Document field."""
    script = await Script.get(script_id)
    if script and script.explicit_cache_name:
        cache_name = script.explicit_cache_name
        await gemini_client.delete_cache(cache_name)
        script.explicit_cache_name = None
        await script.save()
        logger.info("Torn down explicit cache for script %s", script_id)
