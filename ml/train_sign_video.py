"""
Isaret videosu (RGB) -> R3D-18 ince ayari.

Girdi: etiket CSV (örn. etiketlerim_chalearn_val.csv) + eslesen *_color.mp4
Cikti: artifacts/label_map.json, output/sign_r3d/best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models.video import R3D_18_Weights, r3d_18

from sign_label_map import LabelMaps, build_from_csv
from sign_video_dataset import SignVideoDataset


def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", type=Path, default=Path("etiketlerim_chalearn_val.csv"))
    parser.add_argument("--video_dir", type=Path, default=Path("../_data_val/val"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/label_map.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/sign_r3d"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    maps = build_from_csv(args.labels_csv)
    if not maps.stem_to_id:
        raise SystemExit(
            "Hic etiketli ornek yok. CSV'de tr_isaret_veya_cumle veya sinif_id doldurun."
        )

    maps.save(args.artifact)
    full = SignVideoDataset(args.video_dir, maps, num_frames=args.num_frames)
    n = len(full)
    gen = torch.Generator().manual_seed(args.seed)
    if n < 2:
        train_ds, val_ds = full, full
    else:
        n_val = max(1, int(n * args.val_ratio))
        n_val = min(n_val, n - 1)
        n_train = n - n_val
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = maps.num_classes()
    weights_enum = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights_enum)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f}  "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            ckpt = {
                "model": model.state_dict(),
                "num_classes": num_classes,
                "num_frames": args.num_frames,
                "label_map": str(args.artifact.resolve()),
            }
            torch.save(ckpt, args.output_dir / "best.pt")
            print(f"  -> best kaydedildi (val acc {best_acc:.4f})")

    print(f"Tamam. En iyi val dogruluk: {best_acc:.4f}. Cikti: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
