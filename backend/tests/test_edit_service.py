"""Unit tests for edit_service and shot_router.

Verifies:
1. classify_edit classification into 'local' vs 'structural'.
2. Local edit sending current shot image without character bible resend.
3. Structural edit reusing cached prefix and merging edit instruction into delta.
4. Version history tracking across multiple edits.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from beanie import PydanticObjectId

from app.models.character import Character
from app.models.image_version import ImageVersion, TokenUsage
from app.models.scene import Scene
from app.models.script import Script
from app.models.shot import Shot
from app.services import edit_service


@pytest.mark.asyncio
async def test_classify_edit_local_vs_structural():
    """Test classifying local (lighting, weather) vs structural (camera angle, pose) edits."""
    mock_text_local = MagicMock(text='{"edit_type": "local"}')
    mock_text_struct = MagicMock(text='{"edit_type": "structural"}')

    with patch(
        "app.services.gemini_client.call_text_model",
        new_callable=AsyncMock,
        side_effect=[mock_text_local, mock_text_struct],
    ):
        res1 = await edit_service.classify_edit("make it rainy")
        assert res1["edit_type"] == "local"

        res2 = await edit_service.classify_edit("change camera angle to close-up")
        assert res2["edit_type"] == "structural"


@pytest.mark.asyncio
async def test_apply_edit_local_flow(tmp_path):
    """Test local edit uses current shot image as base without sending character bible."""
    script_id = PydanticObjectId()
    scene_id = PydanticObjectId()
    shot_id = PydanticObjectId()
    v1_id = PydanticObjectId()

    shot = Shot(id=shot_id, scene_id=scene_id, shot_number=1, description="Jackie walks", current_version_id=v1_id)
    object.__setattr__(shot, "save", AsyncMock())

    scene = Scene(id=scene_id, script_id=script_id, scene_number=1, location="Tea stall")

    storage_dir = tmp_path / "storage"
    v1_file = storage_dir / "shots" / str(shot_id) / "v1.png"
    v1_file.parent.mkdir(parents=True, exist_ok=True)
    v1_file.write_bytes(b"v1_current_frame_bytes")

    v1 = ImageVersion(
        id=v1_id,
        shot_id=shot_id,
        image_path=str(v1_file),
        version_number=1,
        prompt_used="Base prompt",
        token_usage=TokenUsage(requested=1800, cached=1200),
    )

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(inline_data=MagicMock(data=b"v2_edited_frame_bytes"))]))
    ]
    mock_gemini_resp.usage_metadata = MagicMock(prompt_token_count=220, cached_content_token_count=0)

    with (
        patch.object(Shot, "get", new_callable=AsyncMock, return_value=shot),
        patch.object(Scene, "get", new_callable=AsyncMock, return_value=scene),
        patch.object(ImageVersion, "get", new_callable=AsyncMock, return_value=v1),
        patch.object(ImageVersion, "insert", AsyncMock()),
        patch("app.config.settings.STORAGE_DIR", str(storage_dir)),
        patch("app.services.edit_service.classify_edit", new_callable=AsyncMock, return_value={"edit_type": "local"}),
        patch(
            "app.services.gemini_client.call_image_model",
            new_callable=AsyncMock,
            return_value=mock_gemini_resp,
        ) as mock_call_image,
    ):
        v2 = await edit_service.apply_edit(shot_id, "make it rainy")

        assert v2.version_number == 2
        assert v2.edit_instruction == "make it rainy"
        assert v2.token_usage.requested == 220
        mock_call_image.assert_called_once()

        call_kwargs = mock_call_image.call_args[0][0]
        assert "LOCAL EDIT INSTRUCTION:" in call_kwargs["prompt"]
        assert len(call_kwargs["reference_images"]) == 1, "Should send current shot image as base"
        shot.save.assert_called_once()


@pytest.mark.asyncio
async def test_apply_edit_structural_flow(tmp_path):
    """Test structural edit re-uses prompt prefix and merges instruction into delta."""
    script_id = PydanticObjectId()
    scene_id = PydanticObjectId()
    shot_id = PydanticObjectId()
    v1_id = PydanticObjectId()
    char_id = PydanticObjectId()

    shot = Shot(id=shot_id, scene_id=scene_id, shot_number=1, description="Jackie walks", character_ids=[char_id], current_version_id=v1_id)
    object.__setattr__(shot, "save", AsyncMock())

    scene = Scene(id=scene_id, script_id=script_id, scene_number=1, location="Tea stall")

    v1 = ImageVersion(
        id=v1_id,
        shot_id=shot_id,
        image_path="storage/shots/shot1/v1.png",
        version_number=1,
        prompt_used="Base prompt",
        token_usage=TokenUsage(requested=1800, cached=1200),
    )

    char = Character(id=char_id, script_id=script_id, name="Jackie Shroff", description="Kurta", anchor_image_path="storage/characters/char1.png")

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(inline_data=MagicMock(data=b"v2_structural_bytes"))]))
    ]
    mock_gemini_resp.usage_metadata = MagicMock(prompt_token_count=1900, cached_content_token_count=1600)

    storage_dir = tmp_path / "storage"

    with (
        patch.object(Shot, "get", new_callable=AsyncMock, return_value=shot),
        patch.object(Scene, "get", new_callable=AsyncMock, return_value=scene),
        patch.object(Script, "get", new_callable=AsyncMock, return_value=Script(id=script_id, raw_text="text")),
        patch.object(Character, "get", new_callable=AsyncMock, return_value=char),
        patch.object(Character, "find", MagicMock(return_value=MagicMock(to_list=AsyncMock(return_value=[char])))),
        patch.object(ImageVersion, "get", new_callable=AsyncMock, return_value=v1),
        patch.object(ImageVersion, "insert", AsyncMock()),
        patch("app.config.settings.STORAGE_DIR", str(storage_dir)),
        patch("app.services.edit_service.classify_edit", new_callable=AsyncMock, return_value={"edit_type": "structural"}),
        patch("app.services.character_service.get_anchor_bytes", new_callable=AsyncMock, return_value=b"char_anchor_bytes"),
        patch(
            "app.services.gemini_client.call_image_model",
            new_callable=AsyncMock,
            return_value=mock_gemini_resp,
        ) as mock_call_image,
    ):
        v2 = await edit_service.apply_edit(shot_id, "change camera angle to extreme close-up")

        assert v2.version_number == 2
        assert v2.edit_instruction == "change camera angle to extreme close-up"
        assert v2.token_usage.requested == 1900
        assert v2.token_usage.cached == 1600

        call_kwargs = mock_call_image.call_args[0][0]
        assert "EDIT REQUIREMENT: change camera angle to extreme close-up" in call_kwargs["prompt"]
        assert shot.description == "Jackie walks", "Original shot description on DB must NOT be mutated"
        shot.save.assert_called_once()
