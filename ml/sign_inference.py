"""Video/kare -> Turkce isaret kelimesi (galeri arama + siniflandirici)."""

from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path

_ML = Path(__file__).resolve().parent
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
import torchvision.transforms.functional as VF

from embedding import FrameEmbedder
from gallery_search import VideoGallery
from sign_label_map import LabelMaps
from sign_video_dataset import load_video_clip
from turkish_output import gloss_to_display, gloss_to_sentence
from video_io import read_middle_frame_tensor

_GALLERY_MIN_SIM = 0.38
_LIVE_GALLERY_MIN_SIM = 0.34
_CLASSIFIER_MIN_CONF = 0.12
_LIVE_MIN_FRAMES = 10


def _default_checkpoint() -> Path:
    for p in (
        _ML / "output/sign_frame/best.pt",
        _ML / "output/sign_r3d/best.pt",
    ):
        if p.is_file():
            return p
    return _ML / "output/sign_frame/best.pt"


def _stem_from_filename(name: str | None) -> str | None:
    if not name:
        return None
    s = Path(name).stem
    s = re.sub(r"_color$", "", s, flags=re.IGNORECASE)
    return s or None


class SignInferenceEngine:
    def __init__(self, checkpoint: Path | None = None):
        self.checkpoint_path = Path(checkpoint or _default_checkpoint())
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint yok: {self.checkpoint_path}")

        try:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")

        self.arch = ckpt.get("arch", "r3d18")
        self.num_classes = int(ckpt["num_classes"])
        label_path = Path(ckpt.get("label_map", _ML / "artifacts/label_map.json"))
        self.maps = LabelMaps.load(label_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = FrameEmbedder.from_checkpoint(self.checkpoint_path)

        gallery_path = _ML / "artifacts/gallery.npz"
        self.gallery = VideoGallery(gallery_path) if gallery_path.is_file() else None

        if self.arch == "resnet18_frame":
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.model.to(self.device)
        else:
            from torchvision.models.video import r3d_18

            self.num_frames = int(ckpt.get("num_frames", 16))
            self.model = r3d_18(weights=None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.model.to(self.device)

    def _name_for_id(self, class_id: int) -> str:
        return self.maps.id_to_name.get(str(class_id), f"sinif_{class_id}")

    def _pack(
        self,
        class_id: int,
        confidence: float,
        mode: str,
        alternatives: list[dict] | None = None,
    ) -> dict:
        low = confidence < _GALLERY_MIN_SIM and mode != "filename"
        gloss = self._name_for_id(class_id)
        alts_out: list[dict] = []
        if alternatives:
            for a in alternatives:
                cid = int(a["class_id"])
                alts_out.append(
                    {
                        "word": gloss_to_display(self._name_for_id(cid)),
                        "score": a.get("score", 0),
                    }
                )

        return {
            "raw_text": gloss_to_display(gloss),
            "corrected_text": gloss_to_sentence(gloss, low_confidence=low),
            "class_id": class_id,
            "confidence": round(float(confidence), 4),
            "mode": mode,
            "alternatives": alts_out,
        }

    def _lookup_filename(self, filename: str | None) -> dict | None:
        stem = _stem_from_filename(filename)
        if not stem or stem not in self.maps.stem_to_id:
            return None
        cid = self.maps.stem_to_id[stem]
        return self._pack(cid, 0.99, "filename")

    @torch.no_grad()
    def _classifier_video(self, video_path: Path) -> tuple[int, float]:
        if self.arch == "resnet18_frame":
            frame = read_middle_frame_tensor(video_path, size=224)
            x = frame.unsqueeze(0).to(self.device)
        else:
            x = load_video_clip(video_path, num_frames=self.num_frames, size=112)
            x = x.unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        return int(pred.item()), float(conf.item())

    @torch.no_grad()
    def predict_video_path(self, video_path: Path, filename: str | None = None) -> dict:
        fname = filename or video_path.name
        looked = self._lookup_filename(fname)
        if looked is not None:
            return looked

        emb = self.embedder.embed_video(video_path, n_frames=5)

        if self.gallery is not None:
            cid, conf, alts = self.gallery.search(emb, k=7)
            if conf >= _GALLERY_MIN_SIM:
                return self._pack(cid, conf, "gallery", alts)

        clf_id, clf_conf = self._classifier_video(video_path)
        if clf_conf >= _CLASSIFIER_MIN_CONF:
            return self._pack(clf_id, clf_conf, "classifier")

        if self.gallery is not None:
            cid, conf, alts = self.gallery.search(emb, k=7)
            return self._pack(cid, conf, "gallery_weak", alts)

        return self._pack(clf_id, clf_conf, "classifier")

    @torch.no_grad()
    def predict_frame_bytes(self, data: bytes, filename: str | None = None) -> dict:
        looked = self._lookup_filename(filename)
        if looked is not None:
            return looked

        emb = self.embedder.embed_frame_bytes(data)

        if self.gallery is not None:
            cid, conf, alts = self.gallery.search(emb, k=7)
            if conf >= _GALLERY_MIN_SIM:
                return self._pack(cid, conf, "gallery", alts)

        img = Image.open(__import__("io").BytesIO(data)).convert("RGB")
        frame = VF.to_tensor(img)
        frame = VF.resize(frame, [224, 224], antialias=True)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = ((frame - mean) / std).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        return self._pack(int(pred.item()), float(conf.item()), "classifier")

    def predict_frame_base64(self, b64: str, filename: str | None = None) -> dict:
        if not b64:
            raise ValueError("Bos kare")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        return self.predict_frame_bytes(raw, filename=filename)

    @staticmethod
    def _decode_frame_b64(b64: str) -> bytes:
        s = (b64 or "").strip()
        if not s:
            raise ValueError("Bos kare")
        if "," in s:
            s = s.split(",", 1)[1]
        return base64.b64decode(s)

    @torch.no_grad()
    def _embed_from_frames_b64(self, frames_b64: list[str]) -> torch.Tensor:
        embs: list[torch.Tensor] = []
        for b64 in frames_b64:
            try:
                raw = self._decode_frame_b64(b64)
                embs.append(self.embedder.embed_frame_bytes(raw))
            except (ValueError, OSError):
                continue
        if not embs:
            raise ValueError("Gecerli kare yok")
        stacked = torch.stack(embs, dim=0)
        mean = stacked.mean(dim=0)
        return torch.nn.functional.normalize(mean.unsqueeze(0), dim=1).squeeze(0)

    @torch.no_grad()
    def _live_gallery_vote(self, valid: list[str]) -> tuple[int, float, list[dict]] | None:
        """Her kare icin galeri oyu; tek karede yuksek ama yanlis guveni azaltir."""
        if self.gallery is None:
            return None
        from collections import Counter

        vote_w: Counter[int] = Counter()
        conf_acc: dict[int, float] = {}
        for b64 in valid:
            try:
                raw = self._decode_frame_b64(b64)
                emb1 = self.embedder.embed_frame_bytes(raw)
            except (ValueError, OSError):
                continue
            cid, conf, _ = self.gallery.search(emb1, k=5)
            vote_w[cid] += 1
            conf_acc[cid] = conf_acc.get(cid, 0.0) + conf

        if not vote_w:
            return None
        best_id, _ = vote_w.most_common(1)[0]
        avg_conf = conf_acc[best_id] / vote_w[best_id]
        _, _, alts = self.gallery.search(
            self._embed_from_frames_b64(valid), k=9
        )
        return best_id, avg_conf, alts

    @torch.no_grad()
    def predict_live_frames(self, frames_b64: list[str]) -> dict:
        """Canli yayin: birden fazla kare -> oy + ortalama embedding."""
        valid = [f for f in frames_b64 if f and len(f) > 32]
        if len(valid) < _LIVE_MIN_FRAMES:
            if valid:
                return self.predict_frame_base64(valid[-1])
            raise ValueError(f"En az {_LIVE_MIN_FRAMES} kare gerekli")

        voted = self._live_gallery_vote(valid)
        if voted is not None:
            cid, conf, alts = voted
            if conf >= _LIVE_GALLERY_MIN_SIM:
                return self._pack(cid, conf, "live_gallery", alts)

        emb = self._embed_from_frames_b64(valid)

        if self.gallery is not None:
            cid, conf, alts = self.gallery.search(emb, k=9)
            if conf >= _LIVE_GALLERY_MIN_SIM:
                return self._pack(cid, conf, "live_gallery", alts)

        # Coklu kare siniflandirici oylamasi
        votes: dict[int, float] = {}
        for b64 in valid[-6:]:
            try:
                raw = self._decode_frame_b64(b64)
            except (ValueError, OSError):
                continue
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            frame = VF.to_tensor(img)
            frame = VF.resize(frame, [224, 224], antialias=True)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            x = ((frame - mean) / std).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            c, p = probs.max(dim=1)
            pid = int(p.item())
            votes[pid] = votes.get(pid, 0.0) + float(c.item())

        if votes:
            clf_id = max(votes, key=votes.get)
            clf_conf = votes[clf_id] / len(valid[-6:])
            if self.gallery is not None:
                cid, conf, alts = self.gallery.search(emb, k=9)
                if conf >= clf_conf:
                    return self._pack(cid, conf, "live_gallery_weak", alts)
            return self._pack(clf_id, clf_conf, "live_classifier")

        if self.gallery is not None:
            cid, conf, alts = self.gallery.search(emb, k=9)
            return self._pack(cid, conf, "live_gallery_weak", alts)

        return self.predict_frame_base64(valid[-1])


_engine: SignInferenceEngine | None = None


def get_engine() -> SignInferenceEngine | None:
    global _engine
    if _engine is not None:
        return _engine
    ckpt = _default_checkpoint()
    if not ckpt.is_file():
        return None
    try:
        _engine = SignInferenceEngine(ckpt)
    except Exception:
        return None
    return _engine


def reset_engine() -> None:
    global _engine
    _engine = None
