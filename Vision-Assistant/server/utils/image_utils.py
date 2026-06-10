import cv2
import numpy as np
import base64


def decode_frame(frame_data: str):
    """Decode base64 frame data to numpy array"""

    try:

        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        img_bytes = base64.b64decode(frame_data)

        nparr = np.frombuffer(img_bytes, np.uint8)

        frame = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        return frame

    except Exception as e:

        print(f"Error decoding frame: {e}")

        return None