from fastapi import APIRouter

from schemas.requests import SpeechRequest

from speech import speak_async


router = APIRouter()


@router.post("/speak")
async def speak_endpoint(request: SpeechRequest):

    try:
        speak_async(request.text)

        return {
            "success": True,
            "message": "Speech started"
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }