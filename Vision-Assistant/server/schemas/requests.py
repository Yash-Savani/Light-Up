from pydantic import BaseModel, Field


class FrameRequest(BaseModel):

    frame_data: str = Field(
        ...,
        description="base64 encoded image data"
    )


class SpeechRequest(BaseModel):

    text: str