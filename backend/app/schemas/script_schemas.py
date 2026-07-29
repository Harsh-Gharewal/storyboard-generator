from pydantic import BaseModel
from typing import List, Optional


# ── Requests ──────────────────────────────────────────────────────────

class ScriptParseRequest(BaseModel):
    """Raw script text submitted for parsing."""
    script_text: str
    model: Optional[str] = None


# ── Nested response models ───────────────────────────────────────────

class CharacterOut(BaseModel):
    id: str
    name: str
    description: str
    anchor_image_path: Optional[str] = None


class ShotOut(BaseModel):
    id: str
    shot_number: int
    description: str
    camera_angle: Optional[str] = None
    character_names: List[str] = []
    image_path: Optional[str] = None


class SceneOut(BaseModel):
    id: str
    scene_number: int
    location: str
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    mood: Optional[str] = None
    shots: List[ShotOut] = []


# ── Token Summary model ──────────────────────────────────────────────

class TokenSummaryOut(BaseModel):
    total_prompt_tokens: int = 0
    cached_tokens: int = 0
    fresh_tokens: int = 0
    savings_percentage: float = 0.0
    call_count: int = 0


# ── Top-level responses ──────────────────────────────────────────────

class ScriptParseResponse(BaseModel):
    script_id: str
    scenes: List[SceneOut]
    characters: List[CharacterOut]


class StoryboardGenerateResponse(BaseModel):
    script_id: str
    message: str
    total_shots: int


class ShotStatusOut(BaseModel):
    shot_id: str
    shot_number: int
    scene_number: int
    status: str  # "pending" | "generating" | "completed" | "failed"
    image_path: Optional[str] = None
    error: Optional[str] = None


class StoryboardStatusResponse(BaseModel):
    script_id: str
    status: str  # "pending" | "in_progress" | "completed" | "failed"
    progress: float  # 0.0 – 1.0
    shots: List[ShotStatusOut] = []
    token_summary: Optional[TokenSummaryOut] = None
