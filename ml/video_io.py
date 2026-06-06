"""Video okuma (OpenCV) — torchvision.read_video uyumsuzluklarina karsi."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as VF


def read_middle_frame_tensor(path: Path, size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = max(0, total // 2) if total > 0 else 0
    if target > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Kare okunamadi: {path}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = VF.resize(t, [size, size], antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def _frame_bgr_to_tensor(frame, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = VF.resize(t, [size, size], antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def read_video_frame_tensors(path: Path, n_frames: int = 5, size: int = 224) -> torch.Tensor:
    """Videodan esit aralikli n kare (C,H,W), normalize."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        mid = read_middle_frame_tensor(path, size=size)
        return mid.unsqueeze(0).repeat(n_frames, 1, 1, 1)

    if n_frames <= 1:
        idxs = [total // 2]
    else:
        idxs = [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]

    tensors: list[torch.Tensor] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        tensors.append(_frame_bgr_to_tensor(frame, size))

    cap.release()
    if not tensors:
        raise RuntimeError(f"Kare okunamadi: {path}")
    return torch.stack(tensors, dim=0)
