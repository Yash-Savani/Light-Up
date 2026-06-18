from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, List, Union, Literal
import uvicorn
import json
import time  

from models import load_models, load_known_faces, process_frames, SceneDescriber
from speech import speak_async

# Initialize FastAPI app
app = FastAPI(title="Vision Assistant Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for validation errors (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"\n VALIDATION ERROR (422) ")
    print(f"Request URL: {request.url}")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    
    try:
        body = await request.body()
        if len(body) < 1000:
            print(f"Request body: {body.decode('utf-8')}")
        else:
            print(f"Request body (first 500 chars): {body[:500].decode('utf-8')}...")
    except Exception as e:
        print(f"Could not read request body: {e}")
    
    print(f"Validation errors: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors(),
            "debug_info": {
                "url": str(request.url),
                "method": request.method,
                "error_count": len(exc.errors())
            }
        }
    )

# Global variables to store models and state
object_model = None
face_net = None
shape_predictor = None
face_recognition_model = None
known_face_encodings = []
known_face_names = []
scene_describer = None
last_object_frame_gray = None
last_object_result = None

# Audio state memory
last_speech_time = 0
SPEECH_COOLDOWN = 5.0 # Wait 5 seconds between speaking new things
last_spoken_text = ""  # Memory for the Parrot Filter

class FrameRequest(BaseModel):
    frame_data: str = Field(..., description="base64 encoded image data")
    mode: Literal['face', 'object', 'scene'] = Field(..., description="Processing mode: face, object, or scene")
    confidence: float = Field(0.5, description="Confidence threshold for detections")

class ProcessingResponse(BaseModel):
    detection_result: Optional[Union[List[str], str]] = None
    success: bool = True
    error: Optional[str] = None

class SpeechRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup"""
    global object_model, face_net, shape_predictor, face_recognition_model
    global known_face_encodings, known_face_names, scene_describer
    
    print("Loading models...")
    
    try:
        object_model, face_net, shape_predictor, face_recognition_model = load_models()
        
        print(f"Model loading results:")
        print(f"  object_model: {'✓' if object_model is not None else '✗'}")
        print(f"  face_net: {'✓' if face_net is not None else '✗'}")
        print(f"  shape_predictor: {'✓' if shape_predictor is not None else '✗'}")
        print(f"  face_recognition_model: {'✓' if face_recognition_model is not None else '✗'}")
        
        if all(model is not None for model in [object_model, face_net, shape_predictor, face_recognition_model]):
            known_face_encodings, known_face_names = load_known_faces(
                face_net, shape_predictor, face_recognition_model
            )
            print(f"Loaded {len(known_face_names)} known faces")
            
            scene_describer = SceneDescriber()
            print("All models loaded successfully!")
        else:
            print("Failed to load some models!")
            
    except Exception as e:
        print(f"Error during model loading: {e}")
        import traceback
        traceback.print_exc()

def decode_frame(frame_data: str) -> np.ndarray:
    """Decode base64 frame data to numpy array"""
    try:
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None

@app.post("/process_frame", response_model=ProcessingResponse)
async def process_frame_endpoint(request: FrameRequest):
    """Process a frame based on the specified mode"""
    try:
        if object_model is None or face_net is None or scene_describer is None:
            raise HTTPException(status_code=500, detail="Models not loaded properly")
        
        frame = decode_frame(request.frame_data)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        detection_result = None

        if request.mode == 'face':
            _, current_names = process_frames(
                frame, face_net, shape_predictor, face_recognition_model,
                known_face_encodings, known_face_names
            )
            if current_names:
                detection_result = list(current_names)
            
        elif request.mode == 'object':
            global last_object_frame_gray, last_object_result
            
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)
            
            motion_detected = True
            
            if last_object_frame_gray is not None:
                # STRICTER THRESHOLD: Raised to 50 to ignore shadows and autofocus
                frame_delta = cv2.absdiff(last_object_frame_gray, current_gray)
                thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
                motion_pixels = cv2.countNonZero(thresh)
                
                # LOWERED VOLUME: Lowered to 2% so it catches small objects like a mouse or book
                total_pixels = current_gray.shape[0] * current_gray.shape[1]
                motion_threshold = total_pixels * 0.02 
                
                if motion_pixels < motion_threshold:
                    motion_detected = False
            
            last_object_frame_gray = current_gray
            
            if motion_detected:
                print(f"Processing object detection (conf > {request.confidence})...")
                results = object_model(frame)
                frame_width = frame.shape[1] 
                
                detected_objects = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        
                        # ANTI-HALLUCINATION FIX: Force YOLO to be at least 50% sure
                        conf_score = float(box.conf[0])
                        strict_confidence = max(request.confidence, 0.50)
                        
                        if conf_score > strict_confidence:
                            label = results[0].names[int(box.cls[0])]
                            
                            # IGNORING THE PERSON CLASS
                            # If it's just a person, we completely ignore it and stay silent
                            if label == "person":
                                continue
                            
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            center_x = (x1 + x2) / 2
                            
                            # DIRECTION SWAP FIX: Matches the mirrored selfie-camera perspective
                            if center_x < frame_width / 3:
                                position = "on the right"
                            elif center_x > (2 * frame_width) / 3:
                                position = "on the left"
                            else:
                                position = "in front"
                                
                            detected_objects.append(f"{label} {position}")
                
                if detected_objects:
                    # We found a real object! Return it.
                    last_object_result = list(set(detected_objects))
                else:
                    # We only saw a person (or nothing at all). 
                    # Returning ["nothing"] ensures the TTS engine says the word "nothing" exactly once, 
                    # and then the parrot filter will mute it until an actual object appears.
                    last_object_result = ["nothing"]
                    
                detection_result = last_object_result
                
            else:
                print("No motion detected. Skipping YOLOv8 to save battery.")
                detection_result = last_object_result
            
        elif request.mode == 'scene':
            description = scene_describer.get_scene_description(frame)
            detection_result = description
            
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")
        
        return ProcessingResponse(
            detection_result=detection_result,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return ProcessingResponse(
            detection_result=None,
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "object_model": object_model is not None,
            "face_net": face_net is not None,
            "shape_predictor": shape_predictor is not None,
            "face_recognition_model": face_recognition_model is not None,
            "scene_describer": scene_describer is not None
        },
        "known_faces": len(known_face_names)
    }

@app.post("/speak")
async def speak_endpoint(request: SpeechRequest):
    """Text-to-speech endpoint with a cooldown AND a parrot-filter"""
    global last_speech_time, last_spoken_text
    
    # If the text is empty, do absolutely nothing
    if not request.text or request.text.strip() == "":
        return {"success": False, "message": "No text provided"}

    current_time = time.time()
    
    # 1. PARROT FILTER: Did the scene actually change?
    if request.text == last_spoken_text:
        # If the text is identical, we wait 60 seconds before repeating it
        if current_time - last_speech_time < 60.0:
            print(f"🔇 Muted. (Objects haven't changed. Not repeating.)")
            return {"success": False, "message": "Objects unchanged"}
    
    # 2. STANDARD COOLDOWN: 5 seconds for new objects
    if current_time - last_speech_time >= SPEECH_COOLDOWN:
        try:
            print(f"🔊 Speaking: {request.text}")
            speak_async(request.text)
            
            # Update our memory
            last_speech_time = current_time 
            last_spoken_text = request.text 
            
            return {"success": True, "message": "Speech started"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    else:
        time_left = round(SPEECH_COOLDOWN - (current_time - last_speech_time), 1)
        print(f"🔇 Muted. (Cooldown active for {time_left} more seconds)")
        return {"success": False, "message": "Cooldown active"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)