"""Egitilmis R3D agirliklari ile tek video tahmini."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from sign_label_map import LabelMaps
from sign_video_dataset import load_video_clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True, help="Ornek: ..._color.mp4")
    parser.add_argument("--label_map", type=Path, default=None)
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_classes = int(ckpt["num_classes"])
    num_frames = int(ckpt.get("num_frames", 16))

    label_path = args.label_map or Path(ckpt.get("label_map", "artifacts/label_map.json"))
    maps = LabelMaps.load(label_path)

    weights_enum = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = load_video_clip(args.video, num_frames=num_frames, size=112)
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        pred = int(logits.argmax(dim=1).item())

    name = maps.id_to_name.get(str(pred), str(pred))
    print(f"Tahmin sinif id: {pred}")
    print(f"Etiket adi: {name}")


if __name__ == "__main__":
    main()
