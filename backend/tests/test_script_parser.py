"""Tests for the script_parser service.

Uses a 2-scene sample script about Jackie Shroff promoting Parle-G.
Gemini and MongoDB calls are fully mocked — these tests validate the
parsing, persistence-wiring, and character-reference integrity logic
without hitting any external service.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from beanie import Document, PydanticObjectId

from app.services.script_parser import parse_script


# ── Sample script (2 scenes) ─────────────────────────────────────────

SAMPLE_SCRIPT = """\
SCENE 1 — EXT. VILLAGE TEA STALL — MORNING

A colorful tea stall in a dusty Indian village. JACKIE SHROFF strolls through
the lane in his signature style — sunglasses, white kurta, sleeves rolled up.

A young girl, MEERA (8), sits on a bench looking disappointed at a broken biscuit.

JACKIE
(noticing her)
Arre bachchi, udaas kyun?

He reaches into his pocket and pulls out a bright yellow pack of Parle-G.

JACKIE
Ye le — Parle-G. Genius bana dega!

Meera grabs the pack, breaks a biscuit, and dunks it in her chai.
Her face lights up.


SCENE 2 — EXT. VILLAGE SCHOOL PLAYGROUND — AFTERNOON

The school playground is alive with children. MEERA runs in, waving the pack of
Parle-G, sharing biscuits with her friends.

JACKIE watches from a distance, leaning against a neem tree, smiling.

JACKIE (V.O.)
G maane Genius. Aur Genius ka matlab — Parle-G.

The kids cheer, waving at Jackie. He gives a thumbs up and walks off into
the golden sunlight.
"""


# ── Canned Gemini response matching the sample script ────────────────

MOCK_PARSED = {
    "characters": [
        {
            "name": "Jackie Shroff",
            "description": (
                "Tall Indian man, approximately 65, lean athletic build, "
                "angular face with strong jawline and high cheekbones, "
                "salt-and-pepper hair swept back, medium-brown skin, "
                "wearing black aviator sunglasses and a crisp white "
                "cotton kurta with sleeves rolled to elbows"
            ),
        },
        {
            "name": "Meera",
            "description": (
                "Young Indian girl, approximately 8, petite build, "
                "round face with large brown eyes, black hair in two "
                "braids tied with red ribbons, warm brown skin, wearing "
                "a faded blue cotton school frock"
            ),
        },
    ],
    "scenes": [
        {
            "scene_number": 1,
            "location": "Village tea stall, rural India",
            "time_of_day": "morning",
            "weather": "clear, warm",
            "mood": "cheerful, heartwarming",
            "shots": [
                {
                    "shot_number": 1,
                    "description": "Wide shot of colorful tea stall in dusty village lane.",
                    "camera_angle": "wide establishing, eye level",
                    "characters_present": [],
                    "dialogue": None,
                },
                {
                    "shot_number": 2,
                    "description": "Jackie strolls through lane, sunglasses gleaming.",
                    "camera_angle": "medium tracking shot",
                    "characters_present": ["Jackie Shroff"],
                    "dialogue": None,
                },
                {
                    "shot_number": 3,
                    "description": "Meera sits on bench, looking sadly at broken biscuit.",
                    "camera_angle": "close-up, slight low angle",
                    "characters_present": ["Meera"],
                    "dialogue": None,
                },
                {
                    "shot_number": 4,
                    "description": "Jackie hands Parle-G pack to Meera with warm grin.",
                    "camera_angle": "medium two-shot",
                    "characters_present": ["Jackie Shroff", "Meera"],
                    "dialogue": "Arre bachchi, udaas kyun? Ye le — Parle-G!",
                },
                {
                    "shot_number": 5,
                    "description": "Meera dunks Parle-G biscuit in chai, face lighting up.",
                    "camera_angle": "close-up, eye level",
                    "characters_present": ["Meera"],
                    "dialogue": None,
                },
            ],
        },
        {
            "scene_number": 2,
            "location": "Village school playground",
            "time_of_day": "afternoon",
            "weather": "clear, golden sunlight",
            "mood": "joyful, nostalgic",
            "shots": [
                {
                    "shot_number": 1,
                    "description": "Meera runs into playground sharing Parle-G with friends.",
                    "camera_angle": "wide tracking shot",
                    "characters_present": ["Meera"],
                    "dialogue": None,
                },
                {
                    "shot_number": 2,
                    "description": "Jackie leans against neem tree watching kids, proud smile.",
                    "camera_angle": "medium shot, shallow depth of field",
                    "characters_present": ["Jackie Shroff"],
                    "dialogue": "G maane Genius. Aur Genius ka matlab — Parle-G.",
                },
                {
                    "shot_number": 3,
                    "description": "Kids cheer and wave; Jackie gives thumbs up into sunset.",
                    "camera_angle": "wide shot, golden hour backlighting",
                    "characters_present": ["Jackie Shroff"],
                    "dialogue": None,
                },
            ],
        },
        {
            "scene_number": 3,
            "location": "Neem tree shading the path",
            "time_of_day": "evening",
            "weather": "clear",
            "mood": "peaceful",
            "shots": [
                {
                    "shot_number": 1,
                    "description": "Jackie walks down the dusty path under an orange sunset.",
                    "camera_angle": "wide tracking shot",
                    "characters_present": ["Jackie Shroff"],
                    "dialogue": None,
                },
                {
                    "shot_number": 2,
                    "description": "Jackie stops, turns to camera, and adjusts his sunglasses.",
                    "camera_angle": "medium close-up",
                    "characters_present": ["Jackie Shroff"],
                    "dialogue": None,
                },
                {
                    "shot_number": 3,
                    "description": "Extreme wide shot of Jackie fading into the distance.",
                    "camera_angle": "extreme wide shot",
                    "characters_present": ["Jackie Shroff"],
                    "dialogue": None,
                },
            ],
        },
    ],
}


# ── Fixtures & helpers ───────────────────────────────────────────────

@pytest.fixture
def mock_gemini_response():
    """Create a mock GenerateContentResponse with canned JSON + usage metadata."""
    response = MagicMock()
    response.text = json.dumps(MOCK_PARSED)
    response.usage_metadata = MagicMock()
    response.usage_metadata.prompt_token_count = 450
    response.usage_metadata.cached_content_token_count = 0
    return response


async def _fake_insert(self, *args, **kwargs):
    """Stand-in for Document.insert — assigns a PydanticObjectId without DB."""
    if self.id is None:
        self.id = PydanticObjectId()
    return self


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parse_script_scene_and_shot_counts(mock_gemini_response):
    """Parsed output should have 2 scenes with 5 and 3 shots respectively."""
    with (
        patch(
            "app.services.gemini_client.text_generate",
            new_callable=AsyncMock,
            return_value=mock_gemini_response,
        ),
        patch.object(Document, "insert", _fake_insert),
        patch("app.services.token_logger.log_token_usage"),
    ):
        result = await parse_script(SAMPLE_SCRIPT)

    assert len(result["scenes"]) == 3, "Expected exactly 3 scenes"
    assert len(result["scenes"][0]["shots"]) == 5, "Scene 1 should have 5 shots"
    assert len(result["scenes"][1]["shots"]) == 3, "Scene 2 should have 3 shots"
    assert len(result["scenes"][2]["shots"]) == 3, "Scene 3 should have 3 shots"


@pytest.mark.asyncio
async def test_parse_script_character_count_and_names(mock_gemini_response):
    """Should extract exactly 2 characters: Jackie Shroff and Meera."""
    with (
        patch(
            "app.services.gemini_client.text_generate",
            new_callable=AsyncMock,
            return_value=mock_gemini_response,
        ),
        patch.object(Document, "insert", _fake_insert),
        patch("app.services.token_logger.log_token_usage"),
    ):
        result = await parse_script(SAMPLE_SCRIPT)

    assert len(result["characters"]) == 2
    names = {c["name"] for c in result["characters"]}
    assert names == {"Jackie Shroff", "Meera"}


@pytest.mark.asyncio
async def test_parse_script_characters_present_match_defined(mock_gemini_response):
    """Every character name referenced in shots must match a defined character."""
    with (
        patch(
            "app.services.gemini_client.text_generate",
            new_callable=AsyncMock,
            return_value=mock_gemini_response,
        ),
        patch.object(Document, "insert", _fake_insert),
        patch("app.services.token_logger.log_token_usage"),
    ):
        result = await parse_script(SAMPLE_SCRIPT)

    defined_names = {c["name"] for c in result["characters"]}

    for scene in result["scenes"]:
        for shot in scene["shots"]:
            for name in shot["character_names"]:
                assert name in defined_names, (
                    f"Shot {shot['shot_number']} in scene {scene['scene_number']} "
                    f"references undefined character '{name}'"
                )


@pytest.mark.asyncio
async def test_parse_script_returns_mongo_ids(mock_gemini_response):
    """Every document in the response should have a non-empty string id."""
    with (
        patch(
            "app.services.gemini_client.text_generate",
            new_callable=AsyncMock,
            return_value=mock_gemini_response,
        ),
        patch.object(Document, "insert", _fake_insert),
        patch("app.services.token_logger.log_token_usage"),
    ):
        result = await parse_script(SAMPLE_SCRIPT)

    # Script id
    assert result["script_id"], "script_id must be a non-empty string"

    # Scene and shot ids
    for scene in result["scenes"]:
        assert scene["id"], "scene id must be a non-empty string"
        for shot in scene["shots"]:
            assert shot["id"], "shot id must be a non-empty string"

    # Character ids
    for char in result["characters"]:
        assert char["id"], "character id must be a non-empty string"


@pytest.mark.asyncio
async def test_parse_script_scene_metadata_preserved(mock_gemini_response):
    """Scene-level state (location, time, weather, mood) should be persisted."""
    with (
        patch(
            "app.services.gemini_client.text_generate",
            new_callable=AsyncMock,
            return_value=mock_gemini_response,
        ),
        patch.object(Document, "insert", _fake_insert),
        patch("app.services.token_logger.log_token_usage"),
    ):
        result = await parse_script(SAMPLE_SCRIPT)

    scene1 = result["scenes"][0]
    assert scene1["location"] == "Village tea stall, rural India"
    assert scene1["time_of_day"] == "morning"
    assert scene1["weather"] == "clear, warm"
    assert scene1["mood"] == "cheerful, heartwarming"

    scene2 = result["scenes"][1]
    assert scene2["location"] == "Village school playground"
    assert scene2["time_of_day"] == "afternoon"


@pytest.mark.asyncio
async def test_local_regex_parser_dynamic():
    """Verify that the local regex/rule-based parser parses general screenplays dynamically."""
    custom_script = """
SCENE 1 — EXT. FOREST PATH — MORNING

A dark and mysterious forest path. JOHN walks cautiously, holding a lantern.

A wild wolf jumps out from behind a bush. John drops the lantern.

SCENE 2 — INT. CABIN — NIGHT

John is sitting inside a wooden cabin, bandaging his hand.

MARY sits opposite him, brewing some hot herbal tea.

MARY
Are you alright?

JOHN
I survived.

Mary hands him a warm cup.
"""
    with (
        patch.object(Document, "insert", _fake_insert),
        patch("app.services.token_logger.log_token_usage"),
    ):
        result = await parse_script(custom_script)

    assert len(result["scenes"]) == 2
    assert len(result["characters"]) == 2
    names = {c["name"] for c in result["characters"]}
    assert "John" in names
    assert "Mary" in names

    scene1 = result["scenes"][0]
    assert "FOREST PATH" in scene1["location"].upper()
    assert scene1["time_of_day"] == "morning"
    assert len(scene1["shots"]) >= 3

    scene2 = result["scenes"][1]
    assert "CABIN" in scene2["location"].upper()
    assert scene2["time_of_day"] == "night"
    assert len(scene2["shots"]) >= 3

