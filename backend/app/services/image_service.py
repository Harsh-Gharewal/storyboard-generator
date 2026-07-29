"""Image service — generates storyboard images for shots.

Orchestrates the full shot-image generation flow:
1. Builds prompt with stable prefix (character bible + style guide + scene context)
   from prompt_cache.
2. Loads anchor images ONLY for characters present in the shot.
3. Downscales each reference image to ~768px long edge before sending to API
   to minimize per-call image tokens.
4. Calls gemini-3.1-flash-image via gemini_client.
5. Saves generated image to disk under /storage/shots/{shot_id}/v1.png.
6. Creates ImageVersion document with token usage breakdown.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from beanie import PydanticObjectId

from app.config import settings
from app.models.character import Character
from app.models.image_version import ImageVersion, TokenUsage
from app.models.scene import Scene
from app.models.script import Script
from app.models.shot import Shot
from app.services import character_service, gemini_client, prompt_cache

logger = logging.getLogger(__name__)


def _get_shot_image_path(shot_id: PydanticObjectId, version_number: int = 1) -> Path:
    """Return absolute file path for a shot's generated image version."""
    storage_dir = Path(settings.STORAGE_DIR) / "shots" / str(shot_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir / f"v{version_number}.png"


async def generate_shot_image(
    shot_id: PydanticObjectId,
    *,
    cached_content_name: Optional[str] = None,
) -> ImageVersion:
    """Generate a storyboard image for a single shot.

    Args:
        shot_id: MongoDB ObjectId of the shot.
        cached_content_name: Optional explicit Gemini cache handle.

    Returns:
        Newly created ImageVersion document.
    """
    import time as _time
    t0 = _time.time()

    shot = await Shot.get(shot_id)
    if not shot:
        raise ValueError(f"Shot {shot_id} not found")

    scene = await Scene.get(shot.scene_id)
    if not scene:
        raise ValueError(f"Scene {shot.scene_id} not found for shot {shot_id}")

    # ── SUB-STEP A: Load & downscale reference images ────────────────
    t_ref_start = _time.time()
    
    async def _load_and_process_char(char_id: PydanticObjectId):
        char = await Character.get(char_id)
        if not char:
            return None, None
        anchor_bytes = await character_service.get_anchor_bytes(char.id)
        if not anchor_bytes:
            return char.name, None
        
        downscaled_anchor = await asyncio.to_thread(
            character_service.resize_image_bytes,
            anchor_bytes,
            settings.IMAGE_MAX_LONG_EDGE
        )
        return char.name, downscaled_anchor

    char_tasks = [_load_and_process_char(cid) for cid in shot.character_ids]
    char_results = await asyncio.gather(*char_tasks)

    present_character_names: List[str] = []
    reference_images: List[bytes] = []
    for name, anchor in char_results:
        if name:
            present_character_names.append(name)
        if anchor:
            reference_images.append(anchor)

    t_ref_dur = _time.time() - t_ref_start
    logger.info(
        "    [Shot %s] sub-step A: load refs %.3fs (%d chars, %d images, ~%d KB total)",
        shot_id, t_ref_dur, len(present_character_names), len(reference_images),
        sum(len(b) for b in reference_images) // 1024 if reference_images else 0,
    )

    # ── SUB-STEP B: Build prompt ─────────────────────────────────────
    t_prompt_start = _time.time()
    stable_prefix = await prompt_cache.build_stable_prefix(
        script_id=scene.script_id,
        scene_id=scene.id,
    )

    shot_delta = (
        f"SHOT DELTA:\n"
        f"- Camera Angle: {shot.camera_angle or 'eye-level medium shot'}\n"
        f"- Action: {shot.description}"
    )

    if present_character_names:
        names_str = ", ".join(present_character_names)
        char_instruction = (
            f"REFERENCE INSTRUCTION:\n"
            f"Use the provided reference image(s) as the exact visual appearance for {names_str} — "
            f"preserve face, costume, build, hair, and proportions exactly as shown in the reference image(s)."
        )
    else:
        char_instruction = "REFERENCE INSTRUCTION:\nNo key characters present in this shot; maintain overall visual continuity."

    full_prompt = f"{stable_prefix}\n\n{shot_delta}\n\n{char_instruction}"
    t_prompt_dur = _time.time() - t_prompt_start
    logger.info(
        "    [Shot %s] sub-step B: build prompt %.1fs (%d chars)",
        shot_id, t_prompt_dur, len(full_prompt),
    )

    # ── SUB-STEP C: Call Gemini image model (API call) ────────────────
    t_api_start = _time.time()
    payload = {
        "prompt": full_prompt,
        "reference_images": reference_images if reference_images else None,
        "cached_content_name": cached_content_name,
        "script_id": str(scene.script_id),
        "shot_id": str(shot.id),
    }

    response = await gemini_client.call_image_model(payload)
    t_api_dur = _time.time() - t_api_start
    logger.info(
        "    [Shot %s] sub-step C: Gemini API call %.1fs ⬅ THIS IS THE BOTTLENECK",
        shot_id, t_api_dur,
    )

    # ── SUB-STEP D: Extract image bytes ──────────────────────────────
    t_extract_start = _time.time()
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
        logger.warning("Gemini image response contained no raw image bytes for shot %s; using placeholder", shot_id)
        raw_image_bytes = b"placeholder_shot_image"

    processed_bytes = await asyncio.to_thread(
        character_service.resize_image_bytes,
        raw_image_bytes,
        1024
    )
    t_extract_dur = _time.time() - t_extract_start
    logger.info(
        "    [Shot %s] sub-step D: extract & resize %.3fs (%d KB)",
        shot_id, t_extract_dur, len(processed_bytes) // 1024,
    )

    # ── SUB-STEP E: Save to disk & DB ────────────────────────────────
    t_save_start = _time.time()
    version_num = 1
    existing_versions = await ImageVersion.find(ImageVersion.shot_id == shot.id).to_list()
    if existing_versions:
        version_num = max(v.version_number for v in existing_versions) + 1

    file_path = _get_shot_image_path(shot.id, version_num)
    
    def _write_file():
        with open(file_path, "wb") as f:
            f.write(processed_bytes)
            
    await asyncio.to_thread(_write_file)

    relative_path = f"storage/shots/{shot.id}/v{version_num}.png"

    usage = getattr(response, "usage_metadata", None)
    req_tokens = getattr(usage, "prompt_token_count", 0) or 0
    cached_toks = getattr(usage, "cached_content_token_count", 0) or 0

    image_version = ImageVersion(
        shot_id=shot.id,
        image_path=relative_path,
        version_number=version_num,
        prompt_used=full_prompt,
        token_usage=TokenUsage(requested=req_tokens, cached=cached_toks),
    )
    await image_version.insert()

    # Update shot.current_version_id and status
    shot.current_version_id = image_version.id
    shot.status = "done"
    shot.error = None
    await shot.save()

    t_save_dur = _time.time() - t_save_start
    t_total = _time.time() - t0
    logger.info(
        "    [Shot %s] sub-step E: save to disk/DB %.3fs (tokens: %d req / %d cached)",
        shot_id, t_save_dur, req_tokens, cached_toks,
    )
    logger.info(
        "    [Shot %s] TOTAL: %.3fs (refs=%.3fs + prompt=%.3fs + API=%.3fs + extract=%.3fs + save=%.3fs)",
        shot_id, t_total, t_ref_dur, t_prompt_dur, t_api_dur, t_extract_dur, t_save_dur,
    )

    return image_version


async def generate_scene_shots(
    scene_id: PydanticObjectId,
    *,
    cached_content_name: Optional[str] = None,
) -> list[ImageVersion]:
    """Generate images for all shots in a scene concurrently.

    Args:
        scene_id: MongoDB ObjectId of the scene.
        cached_content_name: Optional explicit Gemini cache handle.

    Returns:
        List of created ImageVersion documents.
    """
    shots = await Shot.find(Shot.scene_id == scene_id).sort("+shot_number").to_list()
    tasks = [
        generate_shot_image(shot.id, cached_content_name=cached_content_name)
        for shot in shots
    ]
    return list(await asyncio.gather(*tasks))


async def generate_all_shots(script_id: PydanticObjectId) -> list[ImageVersion]:
    """Generate storyboard images for every shot in a script concurrently.

    1. Ensures explicit context cache exists via prompt_cache if prefix > threshold.
    2. Generates all missing character anchor images first.
    3. Iterates scenes in scene_number order and generates shots concurrently.

    Args:
        script_id: MongoDB ObjectId of the script.

    Returns:
        List of all created ImageVersion documents across the script.
    """
    # 1. Ensure explicit context cache handle if threshold exceeded
    cached_content_name = await prompt_cache.get_or_create_explicit_cache(script_id)

    # 2. Ensure character anchors exist
    await character_service.generate_all_anchors(script_id)

    # 3. Process scenes in scene_number order concurrently
    scenes = await Scene.find(Scene.script_id == script_id).sort("+scene_number").to_list()
    tasks = [
        generate_scene_shots(scene.id, cached_content_name=cached_content_name)
        for scene in scenes
    ]
    results = await asyncio.gather(*tasks)
    
    all_versions: list[ImageVersion] = []
    for scene_versions in results:
        all_versions.extend(scene_versions)

    logger.info("Completed storyboard generation for script %s (%d total shots)", script_id, len(all_versions))
    return all_versions
