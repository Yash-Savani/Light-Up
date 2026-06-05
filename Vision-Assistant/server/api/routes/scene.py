from fastapi import APIRouter, HTTPException

from schemas.requests import FrameRequest
from schemas.responses import ProcessingResponse

from utils.image_utils import decode_frame

from api.dependencies import model_manager


router = APIRouter()


@router.post(
    "/scene/describe",
    response_model=ProcessingResponse
)
async def process_scene(request: FrameRequest):

    try:
        print("Processing scene description...")

        # Check scene describer
        if model_manager.scene_describer is None:

            raise HTTPException(
                status_code=500,
                detail="Scene describer not loaded"
            )

        # Decode frame
        frame = decode_frame(request.frame_data)

        if frame is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid frame data"
            )

        # Generate description
        description = (
            model_manager.scene_describer
            .get_scene_description(frame)
        )

        print(
            f"Scene description result: {description}"
        )

        return ProcessingResponse(
            detection_result=description,
            success=True
        )

    except HTTPException:
        raise

    except Exception as e:

        print(f"Scene route error: {e}")

        return ProcessingResponse(
            detection_result=None,
            success=False,
            error=str(e)
        )