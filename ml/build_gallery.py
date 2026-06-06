"""Val videolarindan benzerlik arama galerisi (embedding) olusturur."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ML = Path(__file__).resolve().parent
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from embedding import FrameEmbedder
from sign_label_map import LabelMaps, build_from_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", type=Path, default=Path("etiketlerim_chalearn_val.csv"))
    parser.add_argument("--video_dir", type=Path, default=Path("../_data_val/val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("output/sign_frame/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gallery.npz"))
    parser.add_argument("--max_items", type=int, default=2500, help="0 = tum eslesen videolar")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint yok: {args.checkpoint}")

    maps = build_from_csv(args.labels_csv)
    embedder = FrameEmbedder.from_checkpoint(args.checkpoint)

    items: list[tuple[Path, int, str]] = []
    for stem, y in maps.stem_to_id.items():
        p = args.video_dir / f"{stem}_color.mp4"
        if p.is_file():
            items.append((p, y, stem))

    if args.max_items and len(items) > args.max_items:
        items = items[: args.max_items]

    if not items:
        raise SystemExit("Galeri icin video bulunamadi.")

    embs: list[np.ndarray] = []
    labels: list[int] = []
    stems: list[str] = []

    for i, (path, y, stem) in enumerate(items, 1):
        z = embedder.embed_video(path)
        embs.append(z.numpy())
        labels.append(y)
        stems.append(stem)
        if i % 100 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    mat = np.stack(embs, axis=0).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=mat,
        labels=np.array(labels, dtype=np.int32),
        stems=np.array(stems, dtype=object),
    )
    print(f"Galeri: {len(items)} video -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
