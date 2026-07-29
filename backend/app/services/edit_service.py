"""Edit service — applies natural-language edits to existing shot images.

Supports natural-language instructions while preserving character identity and
scene continuity:
- "local": lighting, weather, color grade, background details.
  Sends ONLY the current shot image + instruction text (cheapest call in pipeline,
  no character bible resend).
- "structural": camera angle, pose, framing, composition changes.
  Re-uses cached prefix from prompt_cache, merges edit instruction into shot delta,
  and re-sends character anchors.

Every edit increments version_number, logs token usage, and stores a new ImageVersion.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from beanie import PydanticObjectId
from pydantic import BaseModel

from app.config import settings
from app.models.character import Character
from app.models.image_version import ImageVersion, TokenUsage
from app.models.scene import Scene
from app.models.script import Script
from app.models.shot import Shot
from app.services import character_service, gemini_client, prompt_cache

logger = logging.getLogger(__name__)


# ── Response schema for edit classification ─────────────────────────

class _EditClassificationSchema(BaseModel):
    edit_type: str  # "local" | "structural"


CLASSIFY_SYSTEM_PROMPT = (
    "You are an image edit classifier. Categorize the user's natural language edit "
    "instruction into exactly one of two types:\n"
    "- 'local': changes to lighting, weather, color grade, atmosphere, time of day, "
    "or minor background details where camera angle, character pose, and framing remain intact.\n"
    "- 'structural': changes to camera angle, zoom, framing, character pose, perspective, "
    "or major spatial composition.\n"
    "Output strict JSON matching {\"edit_type\": \"local\" | \"structural\"}."
)


async def classify_edit(instruction: str, script_id: Optional[PydanticObjectId] = None) -> dict[str, str]:
    """Classify a natural-language edit instruction into 'local' or 'structural'.

    Uses gemini-3.5-flash with a terse prompt and structured JSON response schema.

    Args:
        instruction: Natural language edit instruction (e.g. "make it rainy").
        script_id: Optional Script ID to preserve model selection.

    Returns:
        Dict containing {"edit_type": "local" | "structural"}.
    """
    try:
        response = await gemini_client.call_text_model({
            "prompt": f"Instruction: {instruction}",
            "system_instruction": CLASSIFY_SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": _EditClassificationSchema,
            "script_id": str(script_id) if script_id else None,
        })

        parsed = _EditClassificationSchema.model_validate_json(response.text)
        edit_type = parsed.edit_type.lower()
        if edit_type not in ("local", "structural"):
            edit_type = "local"
        logger.info("Classified edit instruction '%s' as '%s'", instruction, edit_type)
        return {"edit_type": edit_type}
    except Exception as e:
        logger.warning("Failed to classify edit '%s' via Gemini (%s); defaulting to 'local'", instruction, e)
        # Rule of thumb heuristic if API call fails
        lower_inst = instruction.lower()
        structural_keywords = ["camera", "angle", "close-up", "close up", "wide shot", "zoom", "pose", "framing", "perspective", "reposition"]
        if any(kw in lower_inst for kw in structural_keywords):
            return {"edit_type": "structural"}
        return {"edit_type": "local"}


async def apply_edit(
    shot_id: PydanticObjectId,
    instruction: str,
    *,
    cached_content_name: Optional[str] = None,
) -> ImageVersion:
    """Apply a natural-language edit to an existing shot image.

    1. Fetches the current ImageVersion for the shot.
    2. Classifies instruction via classify_edit().
    3. If "local": sends current shot image + instruction (no full character bible resend).
    4. If "structural": re-runs prompt construction, merging instruction into shot delta
       and re-sending character anchor images.
    5. Saves new image as ImageVersion (version_number + 1) and updates shot.current_version_id.

    Args:
        shot_id: MongoDB ObjectId of the shot.
        instruction: Edit instruction text.
        cached_content_name: Optional explicit cache handle.

    Returns:
        Newly created ImageVersion document.
    """
    shot = await Shot.get(shot_id)
    if not shot:
        raise ValueError(f"Shot {shot_id} not found")

    if not shot.current_version_id:
        raise ValueError(f"Shot {shot_id} has no generated image version to edit")

    current_version = await ImageVersion.get(shot.current_version_id)
    if not current_version:
        raise ValueError(f"Current ImageVersion {shot.current_version_id} not found")

    scene = await Scene.get(shot.scene_id)
    if not scene:
        raise ValueError(f"Scene {shot.scene_id} not found")

    # 1. Classify edit
    classification = await classify_edit(instruction, script_id=scene.script_id)
    edit_type = classification["edit_type"]

    reference_images: list[bytes] = []
    full_prompt: str = ""

    if edit_type == "local":
        # ── LOCAL EDIT: Cheapest path ─────────────────────────────────
        # Send ONLY the current shot image + instruction text
        current_img_path = Path(current_version.image_path)
        current_bytes: Optional[bytes] = None

        if current_img_path.exists():
            try:
                with open(current_img_path, "rb") as f:
                    current_bytes = f.read()
            except OSError as e:
                logger.warning("Could not read current shot image from %s: %s", current_img_path, e)

        if current_bytes:
            # Downscale current shot image to ~768px long edge
            downscaled_current = character_service.resize_image_bytes(current_bytes, settings.IMAGE_MAX_LONG_EDGE)
            reference_images = [downscaled_current]

            # Log actual dimensions to verify downscaling
            from PIL import Image
            import io
            try:
                with Image.open(io.BytesIO(downscaled_current)) as img:
                    logger.info(
                        "Local edit reference image dimensions: %dx%d (max_long_edge: %d)",
                        img.width,
                        img.height,
                        settings.IMAGE_MAX_LONG_EDGE,
                    )
            except Exception as e:
                logger.warning("Could not read local edit reference image dimensions: %s", e)

        full_prompt = (
            f"LOCAL EDIT INSTRUCTION:\n"
            f"Apply only this change to the provided base frame: {instruction}. "
            f"Do not alter the character's face, identity, costume, or overall composition unless asked."
        )

    else:
        # ── STRUCTURAL EDIT ───────────────────────────────────────────
        # Re-run prompt-building logic (reusing cached prefix)
        # Merge edit instruction into shot description for this call ONLY
        stable_prefix = await prompt_cache.build_stable_prefix(scene.script_id, scene.id)

        merged_description = f"{shot.description}. EDIT REQUIREMENT: {instruction}"
        shot_delta = (
            f"SHOT DELTA:\n"
            f"- Camera Angle: {shot.camera_angle or 'eye-level medium shot'}\n"
            f"- Action & Edit: {merged_description}"
        )

        # Load downscaled anchors (~768px) for present characters
        present_character_names: list[str] = []
        for char_id in shot.character_ids:
            char = await Character.get(char_id)
            if char:
                present_character_names.append(char.name)
                anchor_bytes = await character_service.get_anchor_bytes(char.id)
                if anchor_bytes:
                    downscaled_anchor = character_service.resize_image_bytes(
                        anchor_bytes, settings.IMAGE_MAX_LONG_EDGE
                    )
                    reference_images.append(downscaled_anchor)

                    # Log actual dimensions to verify downscaling
                    from PIL import Image
                    import io
                    try:
                        with Image.open(io.BytesIO(downscaled_anchor)) as img:
                            logger.info(
                                "Structural edit reference image for '%s' dimensions: %dx%d (max_long_edge: %d)",
                                char.name,
                                img.width,
                                img.height,
                                settings.IMAGE_MAX_LONG_EDGE,
                            )
                    except Exception as e:
                        logger.warning("Could not read structural edit reference image dimensions: %s", e)

        if present_character_names:
            names_str = ", ".join(present_character_names)
            char_instruction = (
                f"REFERENCE INSTRUCTION:\n"
                f"Use the provided reference image(s) as the exact visual appearance for {names_str} — "
                f"preserve face, costume, build, and features exactly as shown."
            )
        else:
            char_instruction = ""

        full_prompt = f"{stable_prefix}\n\n{shot_delta}\n\n{char_instruction}"

        # If script has explicit cache, check handle
        if not cached_content_name:
            script = await Script.get(scene.script_id)
            if script and script.explicit_cache_name:
                cached_content_name = script.explicit_cache_name

    logger.info(
        "Applying '%s' edit to Shot %s (v%d -> v%d), prompt_len=%d, ref_images=%d",
        edit_type,
        shot_id,
        current_version.version_number,
        current_version.version_number + 1,
        len(full_prompt),
        len(reference_images),
    )

    # 2. Call gemini-3.1-flash-image
    payload = {
        "prompt": full_prompt,
        "reference_images": reference_images if reference_images else None,
        "cached_content_name": cached_content_name if edit_type == "structural" else None,
        "script_id": str(scene.script_id),
        "shot_id": str(shot.id),
    }

    response = await gemini_client.call_image_model(payload)

    # 3. Extract image bytes
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
        logger.warning("Gemini edit call returned no raw image bytes for shot %s; using placeholder", shot_id)
        raw_image_bytes = b"placeholder_edited_image_bytes"

    processed_bytes = character_service.resize_image_bytes(raw_image_bytes, 1024)

    # 4. Save new image version
    new_version_num = current_version.version_number + 1
    storage_dir = Path(settings.STORAGE_DIR) / "shots" / str(shot.id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    new_file_path = storage_dir / f"v{new_version_num}.png"

    with open(new_file_path, "wb") as f:
        f.write(processed_bytes)

    relative_path = f"storage/shots/{shot.id}/v{new_version_num}.png"

    usage = getattr(response, "usage_metadata", None)
    req_tokens = getattr(usage, "prompt_token_count", 0) or 0
    cached_toks = getattr(usage, "cached_content_token_count", 0) or 0

    new_version = ImageVersion(
        shot_id=shot.id,
        image_path=relative_path,
        version_number=new_version_num,
        prompt_used=full_prompt,
        edit_instruction=instruction,
        token_usage=TokenUsage(requested=req_tokens, cached=cached_toks),
    )
    await new_version.insert()

    # 5. Update shot.current_version_id (without mutating shot.description)
    shot.current_version_id = new_version.id
    await shot.save()

    logger.info(
        "Saved edit ImageVersion v%d for Shot %s to %s (tokens: %d req / %d cached)",
        new_version_num,
        shot_id,
        relative_path,
        req_tokens,
        cached_toks,
    )

    return new_version
