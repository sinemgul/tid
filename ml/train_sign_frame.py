"""
Val renk videolarindan hizli ResNet18 (orta kare) egitimi.
Train arsivi olmadan prototip icin yeterli.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ML = Path(__file__).resolve().parent
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models

from sign_label_map import LabelMaps, build_from_csv
from video_io import read_middle_frame_tensor


class FrameSignDataset(Dataset):
    def __init__(self, video_dir: Path, maps: LabelMaps, size: int = 224):
        self.size = size
        self.items: list[tuple[Path, int]] = []
        for stem, y in maps.stem_to_id.items():
            p = video_dir / f"{stem}_color.mp4"
            if p.is_file():
                self.items.append((p, y))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, y = self.items[idx]
        frame = read_middle_frame_tensor(path, size=self.size)
        return frame, int(y)


def train_epoch(model, loader, optim, criterion, device, epoch: int = 0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        if step % 40 == 0:
            print(f"  epoch {epoch} batch {step}/{len(loader)}", flush=True)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", type=Path, default=Path("etiketlerim_chalearn_val.csv"))
    parser.add_argument("--video_dir", type=Path, default=Path("../_data_val/val"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/label_map.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/sign_frame"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=1200, help="0 = tumu")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    maps = build_from_csv(args.labels_csv)
    maps.save(args.artifact)
    full = FrameSignDataset(args.video_dir, maps)
    items = full.items
    if args.max_samples and len(items) > args.max_samples:
        random.shuffle(items)
        items = items[: args.max_samples]
        full.items = items

    n = len(full)
    if n < 2:
        raise SystemExit("Yeterli video yok.")

    gen = torch.Generator().manual_seed(args.seed)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = maps.num_classes()
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, criterion, device, epoch)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs}  train acc {tr_acc:.3f}  val acc {va_acc:.3f}",
            flush=True,
        )
        if va_acc > best_acc:
            best_acc = va_acc
            ckpt = {
                "model": model.state_dict(),
                "num_classes": num_classes,
                "label_map": str(args.artifact.resolve()),
                "arch": "resnet18_frame",
            }
            torch.save(ckpt, args.output_dir / "best.pt")
            print(f"  -> best.pt (val acc {best_acc:.3f})")

    print(f"Bitti. En iyi val dogruluk: {best_acc:.3f}")


if __name__ == "__main__":
    main()
