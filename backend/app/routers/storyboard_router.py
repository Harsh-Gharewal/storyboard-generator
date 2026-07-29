"""Storyboard router — endpoints for generating and tracking storyboards."""

import asyncio
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from beanie import PydanticObjectId

from app.schemas.script_schemas import (
    StoryboardGenerateResponse,
    StoryboardStatusResponse,
    ShotStatusOut,
    TokenSummaryOut,
)
from app.models.script import Script
from app.models.scene import Scene
from app.models.shot import Shot
from app.models.image_version import ImageVersion
from app.services import image_service, token_logger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/storyboard", tags=["storyboard"])


import time

async def _run_generation_task(script_id: PydanticObjectId) -> None:
    """Background task handler for generating storyboard images sequentially."""
    pipeline_start = time.time()
    try:
        logger.info("=" * 70)
        logger.info("PIPELINE START — script %s", script_id)
        logger.info("=" * 70)
        
        # Import services needed for orchestrated execution
        from app.services import character_service, prompt_cache

        # ── STEP 1: Explicit context cache ───────────────────────────
        step_start = time.time()
        logger.info("[STEP 1/3] Building prompt cache...")
        cached_content_name = await prompt_cache.get_or_create_explicit_cache(script_id)
        step_dur = time.time() - step_start
        logger.info("[STEP 1/3] Prompt cache ready in %.1fs (cache_handle=%s)", step_dur, cached_content_name or "implicit")

        # ── STEP 2: Character anchor images ──────────────────────────
        step_start = time.time()
        logger.info("[STEP 2/3] Generating character anchor images...")
        characters = await character_service.generate_all_anchors(script_id)
        step_dur = time.time() - step_start
        logger.info("[STEP 2/3] Character anchors ready in %.1fs (%d characters)", step_dur, len(characters))

        # ── STEP 3: Shot image generation (concurrent) ───────────────
        step_start = time.time()
        scenes = await Scene.find(Scene.script_id == script_id).sort("+scene_number").to_list()
        total_shots = sum([await Shot.find(Shot.scene_id == s.id).count() for s in scenes])
        logger.info("[STEP 3/3] Starting shot generation — %d scenes, %d total shots (fully concurrent)", len(scenes), total_shots)
        
        shots_done = 0
        shots_failed = 0

        async def _process_shot(scene: Scene, shot: Shot, scene_shots_count: int):
            nonlocal shots_done, shots_failed
            # Skip if already completed
            if shot.status == "done":
                shots_done += 1
                logger.info("  [Scene %d/%d] [Shot %d/%d] SKIP (already done) — progress: %d/%d", scene.scene_number, len(scenes), shot.shot_number, scene_shots_count, shots_done, total_shots)
                return
            
            # Log start of generation
            logger.info("  [Scene %d/%d] [Shot %d/%d] GENERATING... — progress: %d/%d", scene.scene_number, len(scenes), shot.shot_number, scene_shots_count, shots_done, total_shots)
            
            # Update status to "generating"
            shot.status = "generating"
            shot.error = None
            await shot.save()
            
            shot_start = time.time()
            try:
                await image_service.generate_shot_image(shot.id, cached_content_name=cached_content_name)
                duration = time.time() - shot_start
                shots_done += 1
                logger.info("  [Scene %d/%d] [Shot %d/%d] DONE in %.3fs — progress: %d/%d", scene.scene_number, len(scenes), shot.shot_number, scene_shots_count, duration, shots_done, total_shots)
            except Exception as e:
                duration = time.time() - shot_start
                shots_failed += 1
                logger.error("  [Scene %d/%d] [Shot %d/%d] FAILED in %.3fs: %s", scene.scene_number, len(scenes), shot.shot_number, scene_shots_count, duration, e, exc_info=True)
                # Mark shot status as "failed" and save error message
                shot.status = "failed"
                shot.error = str(e)
                await shot.save()

        tasks = []
        for scene in scenes:
            shots = await Shot.find(Shot.scene_id == scene.id).sort("+shot_number").to_list()
            scene_shots_count = len(shots)
            for shot in shots:
                tasks.append(_process_shot(scene, shot, scene_shots_count))
                
        if tasks:
            await asyncio.gather(*tasks)

        step_dur = time.time() - step_start
        total_dur = time.time() - pipeline_start
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE — script %s", script_id)
        logger.info("  Shot generation phase: %.1fs", step_dur)
        logger.info("  Total pipeline time:   %.1fs", total_dur)
        logger.info("  Results: %d done, %d failed, %d total", shots_done, shots_failed, total_shots)
        logger.info("=" * 70)
    except Exception as e:
        total_dur = time.time() - pipeline_start
        logger.error("PIPELINE CRASHED after %.1fs for script %s: %s", total_dur, script_id, e, exc_info=True)
        # Update any pending/generating shots to "failed" so the UI reports the error and stops polling
        try:
            scenes = await Scene.find(Scene.script_id == script_id).to_list()
            for scene in scenes:
                shots = await Shot.find(Shot.scene_id == scene.id).to_list()
                for shot in shots:
                    if shot.status in ("pending", "generating"):
                        shot.status = "failed"
                        shot.error = f"Generation aborted: {e}"
                        await shot.save()
        except Exception as db_err:
            logger.error("Failed to mark shots as failed on generation task abort: %s", db_err)


@router.post("/generate/{script_id}", response_model=StoryboardGenerateResponse)
async def generate_storyboard(
    script_id: str,
) -> StoryboardGenerateResponse:
    """Trigger storyboard image generation for all shots in a script.

    Ensures prompt cache and character anchors are initialized first,
    then launches scene-by-scene shot generation concurrently in the background using asyncio.create_task.
    """
    try:
        obj_id = PydanticObjectId(script_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid script_id format")

    script = await Script.get(obj_id)
    if not script:
        raise HTTPException(status_code=404, detail="Script not found")

    # Count total shots across scenes
    scenes = await Scene.find(Scene.script_id == obj_id).to_list()
    total_shots = 0
    for scene in scenes:
        shots = await Shot.find(Shot.scene_id == scene.id).to_list()
        total_shots += len(shots)

    # Launch generation as a background task using asyncio.create_task
    asyncio.create_task(_run_generation_task(obj_id))

    return StoryboardGenerateResponse(
        script_id=script_id,
        message="Storyboard generation started in background",
        total_shots=total_shots,
    )


@router.get("/{script_id}/status", response_model=StoryboardStatusResponse)
async def get_storyboard_status(script_id: str) -> StoryboardStatusResponse:
    """Get current generation progress and running token summary for a script's storyboard.

    Returns per-shot status (pending/generating/completed/failed), overall progress,
    and token usage breakdown from token_logger.get_summary.
    """
    try:
        obj_id = PydanticObjectId(script_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid script_id format")

    script = await Script.get(obj_id)
    if not script:
        raise HTTPException(status_code=404, detail="Script not found")

    scenes = await Scene.find(
        Scene.script_id == obj_id
    ).sort("+scene_number").to_list()

    shot_statuses: list[ShotStatusOut] = []
    completed = 0
    total = 0

    for scene in scenes:
        shots = await Shot.find(
            Shot.scene_id == scene.id
        ).sort("+shot_number").to_list()

        for shot in shots:
            total += 1
            has_image = shot.current_version_id is not None
            
            # Map Shot's "done" status to the schema's "completed"
            status_val = shot.status or ("completed" if has_image else "pending")
            status_mapped = "completed" if status_val == "done" else status_val

            if has_image:
                completed += 1
                version = await ImageVersion.get(shot.current_version_id)
                image_path = version.image_path if version else None
            else:
                image_path = None

            shot_statuses.append(
                ShotStatusOut(
                    shot_id=str(shot.id),
                    shot_number=shot.shot_number,
                    scene_number=scene.scene_number,
                    status=status_mapped,
                    image_path=image_path,
                    error=shot.error,
                )
            )

    # Determine overall status including possible failure states
    if completed == total and total > 0:
        overall = "completed"
    elif any(s.status == "failed" for s in shot_statuses):
        overall = "failed"
    elif any(s.status in ("generating", "completed") for s in shot_statuses):
        overall = "in_progress"
    else:
        overall = "pending"
    progress = completed / total if total > 0 else 0.0

    # Running token summary for this script
    summary_dict = token_logger.get_summary(script_id)
    token_summary = TokenSummaryOut(
        total_prompt_tokens=summary_dict.get("total_prompt_tokens", 0),
        cached_tokens=summary_dict.get("cached_tokens", 0),
        fresh_tokens=summary_dict.get("fresh_tokens", 0),
        savings_percentage=summary_dict.get("savings_percentage", 0.0),
        call_count=summary_dict.get("call_count", 0),
    )

    return StoryboardStatusResponse(
        script_id=script_id,
        status=overall,
        progress=round(progress, 4),
        shots=shot_statuses,
        token_summary=token_summary,
    )
