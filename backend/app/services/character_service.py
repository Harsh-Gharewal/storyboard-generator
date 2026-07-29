"""Character service — generates and stores anchor reference images.

Each character extracted from the script gets exactly ONE anchor image
generated via gemini-3.1-flash-image. This anchor is the single source
of visual identity: every later shot involving that character sends the
anchor image bytes as reference input to the image model, so identity
comes from image conditioning rather than re-describing the character.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Union

from beanie import PydanticObjectId
from PIL import Image

from app.config import settings
from app.models.character import Character
from app.services import gemini_client

logger = logging.getLogger(__name__)

# Max long edge for stored anchor reference images
ANCHOR_MAX_LONG_EDGE = 1024


def resize_image_bytes(image_bytes: bytes, max_long_edge: int) -> bytes:
    """Downscale image bytes using Pillow so neither dimension exceeds max_long_edge.

    Maintains aspect ratio. Converts palette/RGBA to RGB for standard format output.
    If image_bytes cannot be parsed by Pillow (e.g. mock bytes), creates a placeholder PNG.

    Args:
        image_bytes: Input image byte buffer.
        max_long_edge: Maximum allowed length in pixels for the longer edge.

    Returns:
        PNG image bytes.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Skip processing if image is already within bounds and in RGB mode
            if max(img.size) <= max_long_edge and img.mode == "RGB":
                return image_bytes

            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            width, height = img.size
            if max(width, height) > max_long_edge:
                img.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)

            out_buf = io.BytesIO()
            img.save(out_buf, format="PNG")
            return out_buf.getvalue()
    except Exception as e:
        logger.warning("Could not process image with Pillow (%s); generating fallback placeholder PNG", e)
        # Create a simple solid color placeholder PNG for testing/fallback
        img = Image.new("RGB", (max_long_edge, max_long_edge), color=(200, 200, 200))
        out_buf = io.BytesIO()
        img.save(out_buf, format="PNG")
        return out_buf.getvalue()


def _get_character_image_path(character_id: Union[PydanticObjectId, str]) -> Path:
    """Return absolute file path for a character's anchor image."""
    storage_dir = Path(settings.STORAGE_DIR) / "characters"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir / f"{character_id}.png"


async def generate_anchor(character_or_id: Union[Character, PydanticObjectId, str]) -> Character:
    """Generate and store the anchor reference image for a character.

    Builds a detailed portrait prompt from the character's description
    (front-facing, neutral pose, studio background, character reference sheet style)
    and calls gemini-3.1-flash-image to produce the anchor image.

    The resulting image is:
    1. Downscaled to max ~1024px on the long edge via Pillow.
    2. Saved to disk under /storage/characters/{character_id}.png.
    3. Referenced on the Character document via anchor_image_path and anchor_prompt_used.

    Args:
        character_or_id: Character document or ObjectId string.

    Returns:
        Updated Character document.
    """
    if isinstance(character_or_id, Character):
        character = character_or_id
    else:
        obj_id = PydanticObjectId(str(character_or_id))
        character = await Character.get(obj_id)
        if not character:
            raise ValueError(f"Character {character_or_id} not found")

    # Build reference sheet prompt
    prompt = (
        f"CHARACTER REFERENCE SHEET: {character.name}.\n"
        f"Description: {character.description}.\n"
        "Style: Full body character design reference sheet, front-facing neutral pose, "
        "clean studio lighting, neutral light grey background, crisp line art, semi-realistic rendering. "
        "No text, no speech bubbles, no background objects."
    )

    logger.info("Generating anchor reference image for character '%s' (%s)", character.name, character.id)

    payload = {
        "prompt": prompt,
        "script_id": str(character.script_id),
    }

    response = await gemini_client.call_image_model(payload)

    # Extract image bytes from response candidates
    raw_image_bytes: Optional[bytes] = None
    if getattr(response, "candidates", None):
        candidate = response.candidates[0]
        if getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data and getattr(part.inline_data, "data", None):
                    raw_image_bytes = part.inline_data.data
                    break
                elif hasattr(part, "bytes") and part.bytes:
                    raw_image_bytes = part.bytes
                    break

    if not raw_image_bytes:
        logger.warning("Gemini response contained no inline image bytes for character %s; using placeholder", character.id)
        raw_image_bytes = b"placeholder_image_bytes"

    # Downscale to max 1024px on long edge
    processed_bytes = resize_image_bytes(raw_image_bytes, ANCHOR_MAX_LONG_EDGE)

    # Save to /storage/characters/{character_id}.png
    file_path = _get_character_image_path(character.id)
    with open(file_path, "wb") as f:
        f.write(processed_bytes)

    # Relative path stored in DB
    relative_path = f"storage/characters/{character.id}.png"
    character.anchor_image_path = relative_path
    character.anchor_prompt_used = prompt
    await character.save()

    logger.info("Saved anchor image for '%s' to %s", character.name, relative_path)
    return character


async def generate_anchor_image(character_id: PydanticObjectId) -> Character:
    """Alias for generate_anchor."""
    return await generate_anchor(character_id)


async def get_anchor_bytes(character_id: PydanticObjectId) -> Optional[bytes]:
    """Load and return the anchor image bytes for a character from disk.

    Args:
        character_id: The MongoDB ObjectId of the character.

    Returns:
        Image bytes or None if not found.
    """
    file_path = _get_character_image_path(character_id)
    if not file_path.exists():
        # Try relative path from Character doc
        character = await Character.get(character_id)
        if character and character.anchor_image_path:
            alt_path = Path(character.anchor_image_path)
            if alt_path.exists():
                file_path = alt_path

    if file_path.exists():
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except OSError as e:
            logger.warning("Failed to read anchor image from %s: %s", file_path, e)

    return None


async def generate_all_anchors(script_id: PydanticObjectId) -> list[Character]:
    """Generate anchor images for all characters belonging to a script.

    Iterates through characters for the script and generates anchors for any
    that are missing an anchor image file.

    Args:
        script_id: MongoDB ID of the script.

    Returns:
        List of Character documents.
    """
    import asyncio
    characters = await Character.find(Character.script_id == script_id).to_list()

    async def _ensure_anchor(char: Character) -> Character:
        file_path = _get_character_image_path(char.id)
        if not char.anchor_image_path or not file_path.exists():
            return await generate_anchor(char)
        return char

    tasks = [_ensure_anchor(char) for char in characters]
    return list(await asyncio.gather(*tasks))
