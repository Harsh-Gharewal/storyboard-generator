"""Shot router — endpoints for editing shots and viewing version history."""

from fastapi import APIRouter, HTTPException
from beanie import PydanticObjectId

from app.schemas.shot_schemas import (
    ShotEditRequest,
    ShotEditResponse,
    ShotVersionsResponse,
    ImageVersionOut,
    TokenUsageOut,
    ShotRetryResponse,
)
from app.models.shot import Shot
from app.models.scene import Scene
from app.models.image_version import ImageVersion
from app.services import edit_service, image_service, prompt_cache

router = APIRouter(prefix="/api/shot", tags=["shot"])


@router.post("/{shot_id}/edit", response_model=ShotEditResponse)
async def edit_shot(shot_id: str, request: ShotEditRequest) -> ShotEditResponse:
    """Apply a natural-language edit to an existing shot image.

    Classifies the edit ("local" vs "structural"), sends request to
    gemini-3.1-flash-image, and creates a new ImageVersion.
    Returns both old and new image paths for before/after comparison.
    """
    try:
        obj_id = PydanticObjectId(shot_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shot_id format")

    shot = await Shot.get(obj_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    if not shot.current_version_id:
        raise HTTPException(
            status_code=409,
            detail="Shot has no generated image to edit — generate first",
        )

    # Capture old image path before applying edit
    old_version = await ImageVersion.get(shot.current_version_id)
    old_image_path = old_version.image_path if old_version else None

    try:
        new_version = await edit_service.apply_edit(obj_id, request.instruction)
        return ShotEditResponse(
            shot_id=shot_id,
            old_image_path=old_image_path,
            new_version=_version_to_out(new_version),
        )
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="Shot editing is not yet implemented",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{shot_id}/versions", response_model=ShotVersionsResponse)
async def get_shot_versions(shot_id: str) -> ShotVersionsResponse:
    """Return the full version history for a shot's images."""
    try:
        obj_id = PydanticObjectId(shot_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shot_id format")

    shot = await Shot.get(obj_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    versions = await ImageVersion.find(
        ImageVersion.shot_id == obj_id
    ).sort("+version_number").to_list()

    return ShotVersionsResponse(
        shot_id=shot_id,
        versions=[_version_to_out(v) for v in versions],
    )


def _version_to_out(v: ImageVersion) -> ImageVersionOut:
    """Convert an ImageVersion document to the response schema."""
    return ImageVersionOut(
        id=str(v.id),
        version_number=v.version_number,
        image_path=v.image_path,
        prompt_used=v.prompt_used,
        edit_instruction=v.edit_instruction,
        token_usage=TokenUsageOut(
            requested=v.token_usage.requested,
            cached=v.token_usage.cached,
        ),
        created_at=v.created_at.isoformat(),
    )


@router.post("/{shot_id}/generate", response_model=ShotRetryResponse)
async def generate_shot(shot_id: str) -> ShotRetryResponse:
    """Generate (or re-generate) storyboard image for a single shot."""
    try:
        obj_id = PydanticObjectId(shot_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shot_id format")

    shot = await Shot.get(obj_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    scene = await Scene.get(shot.scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found for shot")

    # Re-use explicit prompt cache if it exists
    cached_content_name = await prompt_cache.get_or_create_explicit_cache(scene.script_id)
    
    # Ensure character anchors exist before generating the shot
    from app.services import character_service
    await character_service.generate_all_anchors(scene.script_id)

    # Set status to generating
    shot.status = "generating"
    shot.error = None
    await shot.save()

    try:
        new_version = await image_service.generate_shot_image(
            obj_id,
            cached_content_name=cached_content_name,
        )
        # generate_shot_image sets status to "done" internally on success
        return ShotRetryResponse(
            shot_id=shot_id,
            status="done",
            image_path=new_version.image_path,
        )
    except Exception as e:
        shot.status = "failed"
        shot.error = str(e)
        await shot.save()
        return ShotRetryResponse(
            shot_id=shot_id,
            status="failed",
            error=str(e),
        )
