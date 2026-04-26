from contextlib import asynccontextmanager
from pathlib import Path
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import onnxruntime as ort

CLASSES = [
    "Corruption", "Crimson", "Desert", "Dungeon", "Forest",
    "Hallow", "Hell", "Jungle", "Mushroom", "Ocean",
    "Snow", "Space", "Underground",
]

MEAN = np.array([0.1473, 0.1647, 0.2079], dtype=np.float32)
STD  = np.array([0.1967, 0.2150, 0.2937], dtype=np.float32)
INPUT_H, INPUT_W = 216, 384
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
LOW_CONFIDENCE_THRESHOLD = 0.40

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "efficientnetv2" / "onnx"
    / "best_efficientnet_v2_s_terraria.onnx"
)

session: ort.InferenceSession | None = None
input_name: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, input_name
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")
    session = ort.InferenceSession(
        str(MODEL_PATH), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    yield
    session = None


app = FastAPI(title="Terraria Biome Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((INPUT_W, INPUT_H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)        # HWC → CHW
    return arr[np.newaxis]              # → (1, 3, H, W)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large. Max 10 MB.")

    try:
        tensor = preprocess(image_bytes)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image. Ensure it is a valid image file.")

    logits = session.run(None, {input_name: tensor})[0][0]  # shape (13,)

    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    predictions = sorted(
        [{"biome": cls, "probability": float(p)} for cls, p in zip(CLASSES, probs)],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return JSONResponse({
        "top_prediction": CLASSES[top_idx],
        "top_probability": top_prob,
        "low_confidence": top_prob < LOW_CONFIDENCE_THRESHOLD,
        "predictions": predictions,
    })
