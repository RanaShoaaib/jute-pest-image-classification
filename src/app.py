import json
import logging
from config import ARTIFACT_DIR
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from inference import load_saved_model, predict, InvalidImageError
from pydantic import BaseModel, Field

MANIFEST_PATH = ARTIFACT_DIR/"manifest.json"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_MB = 5

logger = logging.getLogger(__name__)

class ModelInfoResponse(BaseModel):
    selected_backbone: str
    stage: str
    val_auc_macro: float = Field(ge=0.0, le=1.0)
    test_auc_macro: float = Field(ge=0.0, le=1.0)
    num_classes: int = Field(gt=0)
    class_names: list[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manifest = None
    app.state.model = None

    try:
        with open(MANIFEST_PATH, encoding='utf-8') as f:
            manifest = json.load(f)
        model = load_saved_model(manifest["selected_backbone"], manifest["stage"])
        app.state.manifest = manifest
        app.state.model = model
    except Exception:
        logger.exception("Could not load the deployment model")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", summary="Health Check", description="Check whether the app is running")
def health() -> dict[str, bool | str]:
    if app.state.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    return {"status": "ok", "model_loaded": True}


@app.post("/predict", summary="Predict", description="Predict the label for the uploaded image")
async def make_prediction(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only JPEG and PNG images are supported")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    if app.state.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    manifest = app.state.manifest
    try:
        result = predict(app.state.model, image_bytes, manifest["class_names"])
    except InvalidImageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return result


@app.get("/model/info", summary="Get model info", description="Get information of model used for prediction", response_model=ModelInfoResponse)
def get_model_info():
    if app.state.manifest is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model metadata not loaded")
    return app.state.manifest