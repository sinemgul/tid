"""SignList_ClassId_TR_EN.csv ile etiket CSV'lerindeki tr_isaret_veya_cumle sutununu doldurur."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from two_col_labels_io import load_class_mapping


def enrich(labels_csv: Path, mapping_csv: Path) -> int:
    cmap = load_class_mapping(mapping_csv)
    rows: list[dict[str, str]] = []
    with labels_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV okunamadi")
        fieldnames = list(reader.fieldnames)
        for r in reader:
            rows.append({k: (r.get(k) or "").strip() for k in fieldnames})

    filled = 0
    for r in rows:
        if not r.get("sinif_id"):
            continue
        try:
            cid = int(r["sinif_id"])
        except ValueError:
            continue
        tr = cmap.get(cid, "")
        if tr:
            r["tr_isaret_veya_cumle"] = tr
            filled += 1

    with labels_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return filled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/SignList_ClassId_TR_EN.csv"),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[
            "etiketlerim_chalearn_val.csv",
            "etiketlerim_chalearn_test.csv",
            "etiketlerim_train.csv",
        ],
    )
    args = parser.parse_args()
    for name in args.labels:
        p = Path(name)
        if not p.is_file():
            print(f"Atlandi (yok): {p}")
            continue
        n = enrich(p, args.mapping)
        print(f"{p.name}: {n} satir Turkce dolduruldu")


if __name__ == "__main__":
    main()
