from beanie import Document, PydanticObjectId
from typing import Optional


class Scene(Document):
    """A single scene parsed from the script.

    Groups related shots that share the same location, time of day,
    weather, and mood — these scene-level attributes form part of the
    stable prompt prefix so they land in the cached portion of Gemini
    calls for every shot within the scene.
    """

    script_id: PydanticObjectId
    scene_number: int
    location: str
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    mood: Optional[str] = None

    class Settings:
        name = "scenes"
