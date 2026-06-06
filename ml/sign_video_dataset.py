"""AUTSL benzeri renk videolari -> R3D girdisi (C, T, H, W)."""

from __future__ import annotations

import sys
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

import torch
from torch.utils.data import Dataset
import torchvision.io as tvio
import torchvision.transforms.functional as VF

from sign_label_map import LabelMaps


def load_video_clip(path: Path, num_frames: int = 16, size: int = 112) -> torch.Tensor:
    vid, _, _ = tvio.read_video(str(path), pts_unit="sec", output_format="TCHW")
    if vid.numel() == 0:
        raise RuntimeError(f"Okunamadi veya bos video: {path}")

    t = vid.shape[0]
    idxs = torch.linspace(0, t - 1, num_frames).long()
    clips = vid[idxs].float() / 255.0

    resized: list[torch.Tensor] = []
    for i in range(clips.shape[0]):
        resized.append(VF.resize(clips[i], [size, size], antialias=True))
    x = torch.stack(resized, dim=0)

    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.permute(1, 0, 2, 3)
    return x


class SignVideoDataset(Dataset):
    def __init__(
        self,
        video_dir: Path,
        maps: LabelMaps,
        num_frames: int = 16,
        size: int = 112,
    ):
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.size = size
        self.items: list[tuple[Path, int]] = []
        for stem, y in maps.stem_to_id.items():
            p = self.video_dir / f"{stem}_color.mp4"
            if p.is_file():
                self.items.append((p, y))
        if not self.items:
            raise FileNotFoundError(
                f"Etiketli hic video bulunamadi: {self.video_dir} (color mp4 bekleniyor)"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, y = self.items[idx]
        x = load_video_clip(path, self.num_frames, self.size)
        return x, int(y)
