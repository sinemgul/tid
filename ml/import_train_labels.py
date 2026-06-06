"""
ChaLearn train_labels.csv (basliksiz: kok,ClassId) -> ml/etiketlerim_train.csv

Tum egitim ornekleri tek CSV'de toplanir; sinif_id dolu. Isteg bagli ClassId,TR haritasi
ile Turkce sutunu doldurulur.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

_ML = Path(__file__).resolve().parent
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from two_col_labels_io import load_class_mapping, load_instances  # noqa: E402


def write_etiketlerim_from_labels_csv(
    src: Path,
    output: Path,
    class_mapping: Path | None = None,
) -> tuple[int, int]:
    """(ornek_sayisi, benzersiz_class_sayisi) dondurur."""
    inst = load_instances(src)
    cmap: dict[int, str] = load_class_mapping(class_mapping) if class_mapping else {}

    fieldnames = ["video_koku", "tr_isaret_veya_cumle", "sinif_id", "not"]
    rows: list[dict[str, str]] = []
    for stem in sorted(inst.keys()):
        cid = inst[stem]
        tr = cmap.get(cid, "") if cmap else ""
        rows.append(
            {
                "video_koku": stem,
                "tr_isaret_veya_cumle": tr,
                "sinif_id": str(cid),
                "not": "",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    uclass = len({inst[s] for s in inst})
    return len(rows), uclass


def default_train_labels_path() -> Path | None:
    cands = [Path(p) for p in glob.glob(r"c:/Users/sinem/Desktop/**/train_labels.csv", recursive=True)]
    if not cands:
        return None
    prefers = [p for p in cands if "Yeni klas" in p.as_posix()]
    pool = prefers or cands
    return sorted(pool, key=lambda p: len(str(p)))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_labels",
        type=Path,
        default=None,
        help="Yoksa Masaustu altinda train_labels.csv aranir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("etiketlerim_train.csv"),
    )
    parser.add_argument(
        "--class_mapping",
        type=Path,
        default=None,
        help="ClassId, TR baslikli CSV (varsa Turkce dolar).",
    )
    args = parser.parse_args()

    src = args.train_labels or default_train_labels_path()
    if src is None or not src.is_file():
        raise SystemExit("train_labels.csv bulunamadi. --train_labels ile tam yol verin.")

    n, uclass = write_etiketlerim_from_labels_csv(src, args.output, args.class_mapping)
    print(f"Kaynak: {src}")
    print(f"Ornek: {n}  benzersiz ClassId: {uclass}")
    print(f"Cikti: {args.output.resolve()}")
    if not args.class_mapping:
        print("Not: class_mapping verilmedi; tr_isaret_veya_cumle bos. Harita ekleyince tekrar calistirin.")


if __name__ == "__main__":
    main()
