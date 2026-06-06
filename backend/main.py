import random
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ML_DIR = _REPO_ROOT / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

app = FastAPI(title="Isaret Dili API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    frame: str | None = None


class PredictLiveRequest(BaseModel):
    """Canli kamera: son N kare (base64 JPEG)."""
    frames: list[str] = []


class CorrectRequest(BaseModel):
    text: str


_MOCK_PREDICTIONS: list[tuple[str, str]] = [
    ("MERHABA", "Merhaba."),
    ("BEN OKULA GITMEK", "Ben okula gidiyorum."),
    ("TESSEKKUR", "Teşekkür ederim."),
]


def _model_response(out: dict) -> dict:
    return {
        "raw_text": str(out["raw_text"]),
        "corrected_text": str(out["corrected_text"]),
        "confidence": out.get("confidence"),
        "mode": str(out.get("mode", "model")),
        "alternatives": out.get("alternatives") or [],
    }


def _pick_mock(seed_hint: str | None = None) -> tuple[str, str]:
    if seed_hint:
        idx = sum(ord(ch) for ch in seed_hint) % len(_MOCK_PREDICTIONS)
        return _MOCK_PREDICTIONS[idx]
    return random.choice(_MOCK_PREDICTIONS)


def _get_engine():
    from sign_inference import get_engine

    return get_engine()


@app.get("/health")
def health() -> dict:
    engine = _get_engine()
    gallery = (_ML_DIR / "artifacts/gallery.npz").is_file()
    return {
        "message": "API çalışıyor",
        "model_loaded": engine is not None,
        "gallery_ready": gallery,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    engine = _get_engine()
    if engine is not None and payload.frame:
        try:
            out = engine.predict_frame_base64(payload.frame)
            return _model_response(out)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw, corrected = _pick_mock(payload.frame)
    return {"raw_text": raw, "corrected_text": corrected, "mode": "mock"}


@app.post("/predict-live")
def predict_live(payload: PredictLiveRequest) -> dict:
    """Canli kamera: ardışık kareler (video klibi gibi)."""
    engine = _get_engine()
    if engine is not None and payload.frames:
        try:
            out = engine.predict_live_frames(payload.frames)
            return _model_response(out)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw, corrected = _pick_mock(str(len(payload.frames)))
    return {"raw_text": raw, "corrected_text": corrected, "mode": "mock"}


@app.post("/predict-video")
async def predict_video(file: UploadFile | None = File(None)) -> dict:
    if file is None:
        return {"raw_text": "", "corrected_text": "", "mode": "empty"}

    data = await file.read()
    engine = _get_engine()
    if engine is not None and data:
        suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            out = engine.predict_video_path(tmp_path, filename=file.filename)
            return _model_response(out)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    raw, corrected = _pick_mock(file.filename or "")
    return {"raw_text": raw, "corrected_text": corrected, "mode": "mock"}


@app.post("/llm-correct")
def llm_correct(payload: CorrectRequest) -> dict[str, str]:
    text = (payload.text or "").strip()
    if not text:
        return {"raw_text": "", "corrected_text": ""}
    corrected = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    if corrected[-1] not in ".!?":
        corrected += "."
    return {"raw_text": text, "corrected_text": corrected}


@app.get("/datasets/chalearn")
def chalearn_dataset_status() -> dict:
    names = [
        "etiketlerim_chalearn_val.csv",
        "etiketlerim_chalearn_test.csv",
        "etiketlerim_train.csv",
        "data/SignList_ClassId_TR_EN.csv",
    ]
    out: dict[str, dict] = {}
    for name in names:
        path = _ML_DIR / name
        if not path.is_file():
            out[name] = {"exists": False, "lines": 0}
            continue
        try:
            n = sum(1 for _ in path.open(encoding="utf-8"))
        except OSError:
            n = 0
        out[name] = {"exists": True, "lines": n}
    ckpt = _ML_DIR / "output/sign_frame/best.pt"
    out["sign_frame_checkpoint"] = {
        "exists": ckpt.is_file(),
        "path": str(ckpt),
    }
    return {"ml_dir": str(_ML_DIR), "files": out}
