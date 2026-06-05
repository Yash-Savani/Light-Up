from pydantic import BaseModel
from typing import Optional, List, Union


class ProcessingResponse(BaseModel):

    detection_result: Optional[
        Union[List[str], str]
    ] = None

    success: bool = True

    error: Optional[str] = None