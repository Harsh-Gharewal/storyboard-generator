"""Script router — endpoints for script parsing."""

from fastapi import APIRouter, HTTPException

from app.schemas.script_schemas import ScriptParseRequest, ScriptParseResponse
from app.services import script_parser

router = APIRouter(prefix="/api/script", tags=["script"])


@router.post("/parse", response_model=ScriptParseResponse)
async def parse_script(request: ScriptParseRequest) -> ScriptParseResponse:
    """Parse a raw script into structured scenes, shots, and characters.

    Sends the script text to gemini-3.5-flash for structured extraction,
    persists the results in MongoDB, and returns the full breakdown.
    """
    try:
        result = await script_parser.parse_script(request.script_text, model=request.model)
        return ScriptParseResponse(**result)
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="Script parsing is not yet implemented",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
