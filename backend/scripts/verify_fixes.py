import asyncio
import time
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from beanie import init_beanie, Document, PydanticObjectId
from motor.motor_asyncio import AsyncIOMotorClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.models import ALL_DOCUMENT_MODELS
from app.models.script import Script
from app.models.scene import Scene
from app.models.shot import Shot
from app.services import script_parser, image_service
from app.routers.storyboard_router import _run_generation_task
from app.routers.shot_router import generate_shot as retry_shot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verify_fixes")

# Real 3-scene mock response
THREE_SCENE_MOCK_PARSED = {
    "characters": [
        {"name": "Jackie Shroff", "description": "Tall Indian man, 60s, white kurta, aviator sunglasses."},
        {"name": "Meera", "description": "Young girl, 8, blue school frock, hair in braids."}
    ],
    "scenes": [
        {
            "scene_number": 1,
            "location": "Coffee Shop",
            "time_of_day": "morning",
            "weather": "clear",
            "mood": "cozy",
            "shots": [
                {"shot_number": 1, "description": "Meera sipping coffee.", "camera_angle": "medium shot", "characters_present": ["Meera"]},
                {"shot_number": 2, "description": "Jackie walks in.", "camera_angle": "medium tracking", "characters_present": ["Jackie Shroff"]},
                {"shot_number": 3, "description": "Meera waves at Jackie.", "camera_angle": "over-the-shoulder", "characters_present": ["Jackie Shroff", "Meera"]}
            ]
        },
        {
            "scene_number": 2,
            "location": "City Street",
            "time_of_day": "afternoon",
            "weather": "sunny",
            "mood": "energetic",
            "shots": [
                {"shot_number": 1, "description": "Sidewalk tracking shot.", "camera_angle": "wide tracking", "characters_present": ["Jackie Shroff", "Meera"]},
                {"shot_number": 2, "description": "Jackie points to billboard.", "camera_angle": "medium shot", "characters_present": ["Jackie Shroff"]},
                {"shot_number": 3, "description": "Meera smiles and nods.", "camera_angle": "close-up", "characters_present": ["Meera"]}
            ]
        },
        {
            "scene_number": 3,
            "location": "Art Gallery",
            "time_of_day": "evening",
            "weather": "calm",
            "mood": "inspired",
            "shots": [
                {"shot_number": 1, "description": "Looking at sunset painting.", "camera_angle": "wide establishing", "characters_present": ["Jackie Shroff", "Meera"]},
                {"shot_number": 2, "description": "Jackie smiles warmly.", "camera_angle": "close-up", "characters_present": ["Jackie Shroff"]},
                {"shot_number": 3, "description": "Meera looks inspired.", "camera_angle": "medium shot", "characters_present": ["Meera"]}
            ]
        }
    ]
}

# Helper to generate a placeholder solid color PNG bytes
def get_dummy_png_bytes():
    from PIL import Image
    import io
    img = Image.new("RGB", (768, 768), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

async def main():
    print("=" * 80)
    print(" STARTING VERIFICATION OF ALL THREE FIXES")
    print("=" * 80)

    print("Connecting to MongoDB...")
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    client.append_metadata = lambda *args, **kwargs: None
    await init_beanie(database=client[settings.DATABASE_NAME], document_models=ALL_DOCUMENT_MODELS)
    
    # Setup Mocks for Gemini calls
    mock_text_resp = MagicMock()
    mock_text_resp.text = json.dumps(THREE_SCENE_MOCK_PARSED)
    mock_text_resp.usage_metadata = MagicMock(prompt_token_count=500, cached_content_token_count=0)
    
    dummy_png = get_dummy_png_bytes()
    mock_image_resp = MagicMock()
    mock_image_resp.candidates = [
        MagicMock(content=MagicMock(parts=[MagicMock(inline_data=MagicMock(data=dummy_png))]))
    ]
    mock_image_resp.usage_metadata = MagicMock(prompt_token_count=1500, cached_content_token_count=1000)

    call_text_mock = AsyncMock(return_value=mock_text_resp)
    call_image_mock = AsyncMock(return_value=mock_image_resp)

    patches = [
        patch("app.services.gemini_client.call_text_model", call_text_mock),
        patch("app.services.gemini_client.call_image_model", call_image_mock)
    ]
    for p in patches:
        p.start()

    try:
        # --- FIX 1: Enforce script structure ---
        print("\n--- [FIX 1] Testing Script parsing and constraints validation ---")
        script_text = "Coffee Shop, City Street, Art Gallery screenplay with 3 scenes."
        parse_result = await script_parser.parse_script(script_text)
        script_id = parse_result["script_id"]
        print(f"Parsed Script successfully. Script ID: {script_id}")
        
        scenes = await Scene.find(Scene.script_id == PydanticObjectId(parse_result["script_id"])).sort("+scene_number").to_list()
        print(f"Number of scenes parsed: {len(scenes)}")
        for idx, scene in enumerate(scenes):
            shots = await Shot.find(Shot.scene_id == scene.id).to_list()
            print(f"  Scene {scene.scene_number} ({scene.location}) has {len(shots)} shots")

        # --- FIX 2 & 3: Concurrency, Logging, Sizing, Error tolerance ---
        print("\n--- [FIX 2 & 3] Running Concurrent Generation Task with failure mockup ---")
        
        # Intercept image_service.generate_shot_image to fail exactly one shot
        original_gen_shot_image = image_service.generate_shot_image
        failed_shot_id = None
        
        async def mock_fail_one_shot(shot_id, *args, **kwargs):
            nonlocal failed_shot_id
            # Fail the second shot in Scene 1 to verify execution continues
            shot_obj = await Shot.get(shot_id)
            if shot_obj and shot_obj.shot_number == 2 and failed_shot_id is None:
                failed_shot_id = shot_id
                print(f"[Mock] Intentionally failing generation for Shot {shot_id} to test failure recovery...")
                raise RuntimeError("Mock Gemini API Error: Rate Limit Exceeded")
            return await original_gen_shot_image(shot_id, *args, **kwargs)
            
        image_service.generate_shot_image = mock_fail_one_shot

        start_time = time.time()
        # Run background generation task with PydanticObjectId
        await _run_generation_task(PydanticObjectId(script_id))
        gen_time = time.time() - start_time
        print(f"Total background generation task took: {gen_time:.2f} seconds")

        # Check shot statuses from DB
        all_shots = []
        for s in scenes:
            shots = await Shot.find(Shot.scene_id == s.id).to_list()
            all_shots.extend(shots)

        completed_shots = [s for s in all_shots if s.status == "done"]
        failed_shots = [s for s in all_shots if s.status == "failed"]
        print(f"Shots status summary: completed={len(completed_shots)}, failed={len(failed_shots)}")
        for s in all_shots:
            print(f"  Shot ID {s.id} (Scene {s.scene_id}): status='{s.status}', error='{s.error}'")

        # --- FIX 3: Shot Retry ---
        print("\n--- [FIX 3] Retrying the failed shot via POST /api/shot/{id}/retry ---")
        # Restore original generate_shot_image method so it passes on retry
        image_service.generate_shot_image = original_gen_shot_image
        
        for fs in failed_shots:
            print(f"Retrying failed Shot {fs.id}...")
            retry_res = await retry_shot(str(fs.id))
            print(f"Retry response: status='{retry_res.status}', image_path='{retry_res.image_path}', error='{retry_res.error}'")

        # Re-check status of retried shots
        final_shots = []
        for s in scenes:
            shots = await Shot.find(Shot.scene_id == s.id).to_list()
            final_shots.extend(shots)
            
        final_completed = [s for s in final_shots if s.status == "done"]
        final_failed = [s for s in final_shots if s.status == "failed"]
        print(f"Final status summary: completed={len(final_completed)}, failed={len(final_failed)}")
        
    finally:
        for p in patches:
            p.stop()

if __name__ == "__main__":
    asyncio.run(main())
