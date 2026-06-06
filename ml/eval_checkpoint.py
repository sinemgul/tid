"""
Egitilmis checkpoint ile etiket CSV + video klasorunde basit dogruluk orani.

Ornek:
  python eval_checkpoint.py --checkpoint output/sign_r3d/best.pt ^
    --labels etiketlerim_chalearn_test.csv --video_dir ..\\_data_test\\test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ML = Path(__file__).resolve().parent
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights, r3d_18

from sign_label_map import LabelMaps, build_from_csv
from sign_video_dataset import SignVideoDataset


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--label_map_json", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    num_classes = int(ckpt["num_classes"])
    num_frames = int(ckpt.get("num_frames", 16))

    map_path = args.label_map_json or Path(ckpt.get("label_map", "artifacts/label_map.json"))
    if map_path.is_file():
        maps = LabelMaps.load(map_path)
    else:
        maps = build_from_csv(args.labels)

    ds = SignVideoDataset(args.video_dir, maps, num_frames=num_frames)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    weight_enum = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ok = 0
    tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        ok += (pred == y).sum().item()
        tot += y.numel()

    acc = ok / tot if tot else 0.0
    print(f"Ornek: {tot}  Dogru: {ok}  Dogruluk: {acc:.4f}")


if __name__ == "__main__":
    main()
