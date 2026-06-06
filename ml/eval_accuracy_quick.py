"""Val set uzerinde hizli dogruluk: galeri vs siniflandirici."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ML = Path(__file__).resolve().parent
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from sign_inference import SignInferenceEngine
from sign_label_map import build_from_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", type=Path, default=Path("etiketlerim_chalearn_val.csv"))
    parser.add_argument("--video_dir", type=Path, default=Path("../_data_val/val"))
    parser.add_argument("--n", type=int, default=200, help="Rastgele ornek sayisi")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    maps = build_from_csv(args.labels_csv)
    items: list[tuple[Path, int, str]] = []
    for stem, y in maps.stem_to_id.items():
        p = args.video_dir / f"{stem}_color.mp4"
        if p.is_file():
            items.append((p, y, stem))

    random.seed(args.seed)
    if len(items) > args.n:
        items = random.sample(items, args.n)

    engine = SignInferenceEngine()
    ok_gallery = ok_clf = ok_filename = 0
    n = len(items)

    for path, true_id, stem in items:
        fname_lookup = engine._lookup_filename(f"{stem}_color.mp4")
        if fname_lookup and fname_lookup["class_id"] == true_id:
            ok_filename += 1

        r_g = engine.predict_video_path(path, filename="anonim.mp4")
        if r_g["class_id"] == true_id:
            ok_gallery += 1

        r_c = engine.predict_video_path(path, filename="anonim.mp4")
        # classifier-only: force gallery off by testing _classifier_video
        pred_c, _ = engine._classifier_video(path)
        if pred_c == true_id:
            ok_clf += 1

    print(f"Ornek sayisi: {n}")
    print(f"Dosya adi ile (ust sinir, pratikte yuklemede): {ok_filename/n*100:.1f}%")
    print(f"Galeri arama (anonim isim): {ok_gallery/n*100:.1f}%")
    print(f"Sadece siniflandirici (tek kare): {ok_clf/n*100:.1f}%")


if __name__ == "__main__":
    main()
