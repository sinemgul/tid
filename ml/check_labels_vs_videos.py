"""
Video klasoru (AUTSL: *_color.mp4) ile etiket CSV'deki video_koku sutununu karsilastirir.

Eksik etiket veya fazla (orphan) etiketleri listeler.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def stems_from_videos(video_dir: Path) -> set[str]:
    out: set[str] = set()
    for p in video_dir.glob("*.mp4"):
        name = p.stem
        m = re.match(r"^(?P<stem>.+)_(color|depth)$", name, re.IGNORECASE)
        if m:
            out.add(m.group("stem"))
        else:
            out.add(name)
    return out


def stems_from_csv(labels_csv: Path, column: str) -> set[str]:
    found: set[str] = set()
    with labels_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(f"CSV basliklarinda '{column}' yok. Bulunan: {reader.fieldnames}")
        for row in reader:
            v = (row.get(column) or "").strip()
            if v:
                found.add(v)
    return found


def count_empty_turkish(
    labels_csv: Path,
    video_col: str,
    text_col: str,
    video_stems: set[str],
) -> tuple[int, int]:
    """Videoda olan satirlarda bos tr sayisi ve dolu tr sayisi."""
    empty = 0
    filled = 0
    with labels_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or text_col not in reader.fieldnames:
            raise SystemExit(f"CSV basliklarinda '{text_col}' yok.")
        for row in reader:
            vk = (row.get(video_col) or "").strip()
            if vk not in video_stems:
                continue
            tr = (row.get(text_col) or "").strip()
            if tr:
                filled += 1
            else:
                empty += 1
    return empty, filled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=Path,
        required=True,
        help="Ornek: ..\\_data_val\\val (icinde *_color.mp4 dosyalari)",
    )
    parser.add_argument("--labels_csv", type=Path, required=True)
    parser.add_argument("--video_koku_column", type=str, default="video_koku")
    parser.add_argument(
        "--text_column",
        type=str,
        default="tr_isaret_veya_cumle",
        help="Dolu/bos Turkce etiket istatistigi icin.",
    )
    args = parser.parse_args()

    if not args.video_dir.is_dir():
        raise SystemExit(f"Klasor yok: {args.video_dir}")

    v_stems = stems_from_videos(args.video_dir)
    c_stems = stems_from_csv(args.labels_csv, args.video_koku_column)

    missing_labels = sorted(v_stems - c_stems)
    orphan_labels = sorted(c_stems - v_stems)

    print(f"Videoda bulunan kok sayisi: {len(v_stems)}")
    print(f"CSV'de kok sayisi: {len(c_stems)}")
    print(f"Etiketi olmayan video kokleri: {len(missing_labels)}")
    if missing_labels:
        for s in missing_labels[:50]:
            print(f"  - {s}")
        if len(missing_labels) > 50:
            print(f"  ... ve {len(missing_labels) - 50} tane daha")
    print(f"Videoda karsiligi olmayan CSV satirlari: {len(orphan_labels)}")
    if orphan_labels:
        for s in orphan_labels[:50]:
            print(f"  - {s}")
        if len(orphan_labels) > 50:
            print(f"  ... ve {len(orphan_labels) - 50} tane daha")

    if not missing_labels and not orphan_labels:
        print("Tam eslesme: tum videolarin etiketi var, fazla etiket yok.")

    empty_tr, filled_tr = count_empty_turkish(
        args.labels_csv,
        args.video_koku_column,
        args.text_column,
        v_stems,
    )
    print(f"Turkce metin bos satir (video ile eslesen): {empty_tr}")
    print(f"Turkce metin dolu satir (video ile eslesen): {filled_tr}")


if __name__ == "__main__":
    main()
