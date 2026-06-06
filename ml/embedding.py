"""ResNet18 goruntu embedding (sinif katmani oncesi)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from video_io import read_middle_frame_tensor, read_video_frame_tensors


class FrameEmbedder:
    def __init__(self, backbone: nn.Module, device: torch.device):
        self.backbone = backbone
        self.device = device
        self.backbone.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> "FrameEmbedder":
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location="cpu")

        num_classes = int(ckpt["num_classes"])
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(ckpt["model"])
        backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone.to(device)
        return cls(backbone, device)

    @torch.no_grad()
    def _embed_batch(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.to(self.device)
        z = self.backbone(x)
        z = torch.nn.functional.normalize(z, dim=1)
        return z.cpu()

    @torch.no_grad()
    def embed_tensor(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        return self._embed_batch(frame)[0]

    @torch.no_grad()
    def embed_video(self, path: Path, n_frames: int = 5) -> torch.Tensor:
        frames = read_video_frame_tensors(path, n_frames=n_frames, size=224)
        z = self._embed_batch(frames)
        z = torch.nn.functional.normalize(z.mean(dim=0, keepdim=True), dim=1)
        return z[0]

    @torch.no_grad()
    def embed_frame_bytes(self, data: bytes) -> torch.Tensor:
        import io

        from PIL import Image
        import torchvision.transforms.functional as VF

        img = Image.open(io.BytesIO(data)).convert("RGB")
        frame = VF.to_tensor(img)
        frame = VF.resize(frame, [224, 224], antialias=True)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = (frame - mean) / std
        return self.embed_tensor(frame)
