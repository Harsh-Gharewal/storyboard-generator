from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional


class TokenUsage(BaseModel):
    """Token counts from a single Gemini API call."""

    requested: int = 0
    cached: int = 0


class ImageVersion(Document):
    """A versioned image generated for a shot.

    Every generation or edit produces a new ImageVersion so the user
    can view the full history. token_usage records the requested vs.
    cached token counts from the Gemini response for measurable
    savings tracking.
    """

    shot_id: PydanticObjectId
    image_path: str
    version_number: int
    prompt_used: str
    edit_instruction: Optional[str] = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "image_versions"
