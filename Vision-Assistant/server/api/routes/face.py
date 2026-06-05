from fastapi import APIRouter, HTTPException

from schemas.requests import FrameRequest
from schemas.responses import ProcessingResponse

from utils.image_utils import decode_frame

from services.face_service import recognize_faces

from api.dependencies import model_manager


router = APIRouter()


@router.post(
    "/face/recognize",
    response_model=ProcessingResponse
)
async def recognize_face_endpoint(request: FrameRequest):

    try:

        print("Processing face recognition...")

        # Check models
        if model_manager.face_net is None:

            raise HTTPException(
                status_code=500,
                detail="Face recognition models not loaded"
            )

        # Decode frame
        frame = decode_frame(request.frame_data)

        if frame is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid frame data"
            )

        # Face recognition
        _, current_names = recognize_faces(
            frame,
            model_manager.face_net,
            model_manager.shape_predictor,
            model_manager.face_recognition_model,
            model_manager.known_face_encodings,
            model_manager.known_face_names
        )

        detection_result = list(current_names)

        print(f"Face recognition result: {detection_result}")

        return ProcessingResponse(
            detection_result=detection_result,
            success=True
        )

    except HTTPException:
        raise

    except Exception as e:

        print(f"Face route error: {e}")

        return ProcessingResponse(
            detection_result=None,
            success=False,
            error=str(e)
        )