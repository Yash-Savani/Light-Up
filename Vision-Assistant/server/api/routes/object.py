from fastapi import APIRouter, HTTPException

from schemas.requests import FrameRequest
from schemas.responses import ProcessingResponse

from utils.image_utils import decode_frame

from api.dependencies import model_manager

from services.object_service import detect_objects


router = APIRouter()


@router.post(
    "/object/detect",
    response_model=ProcessingResponse
)
async def process_object(request: FrameRequest):

    try:
        print("Processing object detection...")

        # Check model
        if model_manager.object_model is None:

            raise HTTPException(
                status_code=500,
                detail="Object model not loaded"
            )

        # Decode frame
        frame = decode_frame(request.frame_data)

        if frame is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid frame data"
            )

        # Detect objects
        detection_result = detect_objects(
            frame,
            model_manager.object_model
        )

        print(
            f"Object detection result: {detection_result}"
        )

        return ProcessingResponse(
            detection_result=detection_result,
            success=True
        )

    except HTTPException:
        raise

    except Exception as e:

        print(f"Object route error: {e}")

        return ProcessingResponse(
            detection_result=None,
            success=False,
            error=str(e)
        )