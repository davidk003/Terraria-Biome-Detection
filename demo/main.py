from contextlib import asynccontextmanager
from pathlib import Path
import base64
import io
import tempfile

import cv2
import gdown
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
MAX_FILE_BYTES = 10 * 1024 * 1024    # 10 MB
MAX_VIDEO_BYTES = 200 * 1024 * 1024  # 200 MB
LOW_CONFIDENCE_THRESHOLD = 0.40
VIDEO_SAMPLE_FRAMES = 20

DEMO_VIDEO_FILE_ID = "10WWZH_-g83VE8rdrb0Zh2HsdbeZWZ5Hq"
DEMO_VIDEO_CACHE = Path(__file__).parent / "demo_video_cache.mp4"

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


def _infer_pil(img: Image.Image) -> tuple[str, float]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    logits = session.run(None, {input_name: preprocess(buf.getvalue())})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    top = int(np.argmax(probs))
    return CLASSES[top], float(probs[top])


def extract_frame_predictions(video_path: Path, max_frames: int = VIDEO_SAMPLE_FRAMES) -> list:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        raise ValueError("Could not read any frames from the video.")

    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    results = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        biome, prob = _infer_pil(pil_img)

        # 192×108 thumbnail encoded as base64 JPEG
        thumb = pil_img.resize((192, 108), Image.LANCZOS)
        tbuf = io.BytesIO()
        thumb.save(tbuf, format="JPEG", quality=72)
        thumb_b64 = base64.b64encode(tbuf.getvalue()).decode()

        t = idx / fps
        mins, secs = divmod(int(t), 60)
        results.append({
            "timestamp": round(t, 1),
            "timestamp_str": f"{mins}:{secs:02d}",
            "top_prediction": biome,
            "top_probability": prob,
            "thumbnail": f"data:image/jpeg;base64,{thumb_b64}",
        })

    cap.release()
    return results


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


@app.get("/predict-demo")
async def predict_demo():
    if not DEMO_VIDEO_CACHE.exists():
        try:
            gdown.download(id=DEMO_VIDEO_FILE_ID, output=str(DEMO_VIDEO_CACHE), quiet=True)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not download demo video: {exc}")

    try:
        frames = extract_frame_predictions(DEMO_VIDEO_CACHE)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}")

    return JSONResponse({"frames": frames})


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    ct = file.content_type or ""
    if not (ct.startswith("video/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=400, detail="File must be a video (MP4, WebM, etc.)")

    video_bytes = await file.read()
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video too large. Max 200 MB.")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)

    try:
        frames = extract_frame_predictions(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process video: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse({"frames": frames})
