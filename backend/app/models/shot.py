from beanie import Document, PydanticObjectId
from typing import List, Optional


class Shot(Document):
    """A single shot within a scene.

    References the characters present (by ID) so that only the relevant
    anchor images are sent to the Gemini image-generation call.
    current_version_id points to the latest ImageVersion for this shot.
    """

    scene_id: PydanticObjectId
    shot_number: int
    description: str
    camera_angle: Optional[str] = None
    character_ids: List[PydanticObjectId] = []
    current_version_id: Optional[PydanticObjectId] = None
    status: str = "pending"  # "pending" | "generating" | "done" | "failed"
    error: Optional[str] = None

    class Settings:
        name = "shots"
