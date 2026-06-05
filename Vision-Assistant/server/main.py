from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import uvicorn

from api.dependencies import model_manager

from api.routes.face import router as face_router
from api.routes.object import router as object_router
from api.routes.scene import router as scene_router
from api.routes.speech import router as speech_router
from api.routes.health import router as health_router


app = FastAPI(
    title="Vision Assistant Server",
    version="1.0.0"
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Validation handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors()
        }
    )


# Startup
@app.on_event("startup")
async def startup_event():

    try:

        model_manager.initialize_all()

    except Exception as e:

        print(f"Startup error: {e}")

        import traceback
        traceback.print_exc()


# Routers
app.include_router(face_router)
app.include_router(object_router)
app.include_router(scene_router)
app.include_router(speech_router)
app.include_router(health_router)


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )