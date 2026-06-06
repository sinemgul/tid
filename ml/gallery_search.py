"""Embedding galerisinde k-NN arama."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch


class VideoGallery:
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=True)
        self.embeddings = torch.from_numpy(data["embeddings"]).float()
        self.labels = data["labels"].astype(int)
        self.stems = list(data["stems"])
        self.embeddings = torch.nn.functional.normalize(self.embeddings, dim=1)

    def search(self, query: torch.Tensor, k: int = 7) -> tuple[int, float, list[dict]]:
        q = torch.nn.functional.normalize(query.float().unsqueeze(0), dim=1)
        sims = (q @ self.embeddings.T).squeeze(0)
        k = min(k, sims.numel())
        vals, idxs = torch.topk(sims, k=k)

        votes: Counter[int] = Counter()
        for i, s in zip(idxs.tolist(), vals.tolist()):
            votes[int(self.labels[i])] += float(s)

        best_id, score = votes.most_common(1)[0]
        confidence = float(vals[0].item())

        alts: list[dict] = []
        seen: set[int] = set()
        for i, s in zip(idxs.tolist(), vals.tolist()):
            cid = int(self.labels[i])
            if cid in seen:
                continue
            seen.add(cid)
            alts.append({"class_id": cid, "score": round(float(s), 4)})
            if len(alts) >= 3:
                break

        return best_id, confidence, alts
