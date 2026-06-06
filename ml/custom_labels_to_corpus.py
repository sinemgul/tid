"""
Manuel veya yarim-otomatik etiket CSV -> dil modeli korpusu (satir basi metin).

Beklenen UTF-8 CSV (virgul ile, baslik satiri zorunlu):
  video_koku, tr_isaret_veya_cumle, ...

varsayilan metin sutunu: tr_isaret_veya_cumle
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", type=Path, required=True)
    parser.add_argument("--text_column", type=str, default="tr_isaret_veya_cumle")
    parser.add_argument("--output", type=Path, default=Path("corpus_from_labels.txt"))
    args = parser.parse_args()

    rows: list[str] = []
    with args.labels_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or args.text_column not in reader.fieldnames:
            raise SystemExit(
                f"CSV basliklarinda '{args.text_column}' yok. Bulunan: {reader.fieldnames}"
            )
        for rec in reader:
            t = (rec.get(args.text_column) or "").strip()
            if t:
                rows.append(t)

    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Satir: {len(rows)} -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
