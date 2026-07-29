from beanie import Document
from pydantic import Field
from datetime import datetime, timezone
from typing import Optional


class Script(Document):
    """The raw script text uploaded by the user.

    Stores the original text and, once built, the name/handle of the
    Gemini explicit context cache so subsequent image-generation calls
    can reference it instead of re-sending the full character bible
    and style guide as plain text.
    """

    raw_text: str
    explicit_cache_name: Optional[str] = None
    model: Optional[str] = "gemini-3.5-flash"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "scripts"
