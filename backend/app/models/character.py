from beanie import Document, PydanticObjectId
from typing import Optional


class Character(Document):
    """A character extracted from the script.

    Each character gets exactly one anchor reference image, generated once
    and reused as image-conditioning input for every subsequent shot
    featuring that character.
    """

    script_id: PydanticObjectId
    name: str
    description: str
    anchor_image_path: Optional[str] = None
    anchor_prompt_used: Optional[str] = None

    class Settings:
        name = "characters"
