from pydantic import BaseModel
from typing import List, Optional


# ── Requests ──────────────────────────────────────────────────────────

class ShotEditRequest(BaseModel):
    """Natural-language instruction to edit an existing shot image."""
    instruction: str


# ── Nested response models ───────────────────────────────────────────

class TokenUsageOut(BaseModel):
    requested: int
    cached: int


class ImageVersionOut(BaseModel):
    id: str
    version_number: int
    image_path: str
    prompt_used: str
    edit_instruction: Optional[str] = None
    token_usage: TokenUsageOut
    created_at: str


# ── Top-level responses ──────────────────────────────────────────────

class ShotVersionsResponse(BaseModel):
    shot_id: str
    versions: List[ImageVersionOut]


class ShotEditResponse(BaseModel):
    shot_id: str
    old_image_path: Optional[str] = None
    new_version: ImageVersionOut


class ShotRetryResponse(BaseModel):
    shot_id: str
    status: str
    image_path: Optional[str] = None
    error: Optional[str] = None
