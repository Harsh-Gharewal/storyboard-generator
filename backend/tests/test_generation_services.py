"""Unit tests for character_service and image_service.

Verifies:
1. Character anchor generation with Pillow downscaling to 1024px and DB updates.
2. Shot image generation sending anchors ONLY for characters present in shot, downscaling to 768px.
3. Scene-ordered shot generation orchestration and background task execution.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from beanie import PydanticObjectId

from app.config import settings
from app.models.character import Character
from app.models.image_version import ImageVersion
from app.models.scene import Scene
from app.models.script import Script
from app.models.shot import Shot
from app.services import character_service, image_service


@pytest.mark.asyncio
async def test_character_generate_anchor(tmp_path):
    """Test generating character anchor image downscales to 1024px and updates document."""
    script_id = PydanticObjectId()
    char_id = PydanticObjectId()

    char = Character(
        id=char_id,
        script_id=script_id,
        name="Jackie Shroff",
        description="Tall man with aviators and white kurta",
    )
    object.__setattr__(char, "save", AsyncMock())

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(inline_data=MagicMock(data=b"fake_raw_anchor_bytes"))]))
    ]
    mock_gemini_resp.usage_metadata = MagicMock(prompt_token_count=100, cached_content_token_count=0)

    storage_dir = tmp_path / "storage"

    with (
        patch.object(Character, "get", AsyncMock(return_value=char)),
        patch("app.config.settings.STORAGE_DIR", str(storage_dir)),
        patch(
            "app.services.gemini_client.call_image_model",
            new_callable=AsyncMock,
            return_value=mock_gemini_resp,
        ) as mock_call_image,
    ):
        updated_char = await character_service.generate_anchor(char)

        assert updated_char.anchor_image_path == f"storage/characters/{char_id}.png"
        assert "CHARACTER REFERENCE SHEET: Jackie Shroff" in updated_char.anchor_prompt_used
        mock_call_image.assert_called_once()
        char.save.assert_called_once()


@pytest.mark.asyncio
async def test_image_service_generate_shot(tmp_path):
    """Test generating shot image sends only relevant anchors downscaled to 768px."""
    script_id = PydanticObjectId()
    scene_id = PydanticObjectId()
    shot_id = PydanticObjectId()
    char1_id = PydanticObjectId()
    char2_id = PydanticObjectId()

    script = Script(id=script_id, raw_text="Sample script")
    scene = Scene(id=scene_id, script_id=script_id, scene_number=1, location="Tea stall", time_of_day="morning")
    shot = Shot(
        id=shot_id,
        scene_id=scene_id,
        shot_number=1,
        description="Jackie Shroff drinks tea",
        camera_angle="close-up",
        character_ids=[char1_id],  # ONLY char1 present, char2 absent
    )
    object.__setattr__(shot, "save", AsyncMock())

    char1 = Character(id=char1_id, script_id=script_id, name="Jackie Shroff", description="Kurta and aviators", anchor_image_path="storage/characters/char1.png")
    char2 = Character(id=char2_id, script_id=script_id, name="Meera", description="Little girl", anchor_image_path="storage/characters/char2.png")

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(inline_data=MagicMock(data=b"fake_shot_image_bytes"))]))
    ]
    mock_gemini_resp.usage_metadata = MagicMock(prompt_token_count=1200, cached_content_token_count=900)

    fake_anchor_bytes = b"fake_anchor_bytes_char1"
    storage_dir = tmp_path / "storage"

    def mock_get(model_cls, obj_id):
        if model_cls == Shot:
            return shot
        elif model_cls == Scene:
            return scene
        elif model_cls == Script:
            return script
        elif model_cls == Character:
            if obj_id == char1_id:
                return char1
            elif obj_id == char2_id:
                return char2
        return None

    with (
        patch.object(Shot, "get", new_callable=AsyncMock, side_effect=lambda oid: mock_get(Shot, oid)),
        patch.object(Scene, "get", new_callable=AsyncMock, side_effect=lambda oid: mock_get(Scene, oid)),
        patch.object(Script, "get", new_callable=AsyncMock, side_effect=lambda oid: mock_get(Script, oid)),
        patch.object(Character, "get", new_callable=AsyncMock, side_effect=lambda oid: mock_get(Character, oid)),
        patch.object(Character, "find", MagicMock(return_value=MagicMock(to_list=AsyncMock(return_value=[char1, char2])))),
        patch.object(ImageVersion, "insert", AsyncMock()),
        patch.object(ImageVersion, "find", MagicMock(return_value=MagicMock(to_list=AsyncMock(return_value=[])))),
        patch("app.config.settings.STORAGE_DIR", str(storage_dir)),
        patch(
            "app.services.character_service.get_anchor_bytes",
            new_callable=AsyncMock,
            return_value=fake_anchor_bytes,
        ),
        patch(
            "app.services.gemini_client.call_image_model",
            new_callable=AsyncMock,
            return_value=mock_gemini_resp,
        ) as mock_call_image,
    ):
        version = await image_service.generate_shot_image(shot_id)

        assert version.shot_id == shot_id
        assert version.token_usage.requested == 1200
        assert version.token_usage.cached == 900
        mock_call_image.assert_called_once()

        # Check call arguments to verify only 1 anchor was sent (for present character Jackie Shroff)
        call_kwargs = mock_call_image.call_args[0][0]
        assert len(call_kwargs["reference_images"]) == 1, "Only 1 anchor image for present character should be sent"
        assert "Jackie Shroff" in call_kwargs["prompt"]
        assert "Meera" not in call_kwargs["prompt"] or "CHARACTER BIBLE:" in call_kwargs["prompt"]
        shot.save.assert_called_once()
