from fastapi import APIRouter

from api.dependencies import model_manager


router = APIRouter()


@router.get("/health")
async def health_check():

    return {
        "status": "healthy",

        "models_loaded": {

            "object_model":
                model_manager.object_model is not None,

            "face_net":
                model_manager.face_net is not None,

            "shape_predictor":
                model_manager.shape_predictor is not None,

            "face_recognition_model":
                model_manager.face_recognition_model is not None,

            "scene_describer":
                model_manager.scene_describer is not None
        },

        "known_faces":
            len(model_manager.known_face_names)
    }