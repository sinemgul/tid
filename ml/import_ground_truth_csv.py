"""
ChaLearn validation_labels.zip / test_labels.zip icindeki ground_truth.csv
(basliksiz: kok,ClassId) -> etiketlerim_val.csv / etiketlerim_test.csv

Once zip'i dogru sifre ile cikartin; sonra:
  python import_ground_truth_csv.py --ground_truth yol\\ground_truth.csv --output etiketlerim_val.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from import_train_labels import write_etiketlerim_from_labels_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class_mapping", type=Path, default=None)
    args = parser.parse_args()

    if not args.ground_truth.is_file():
        raise SystemExit(f"Dosya yok: {args.ground_truth}")

    n, u = write_etiketlerim_from_labels_csv(
        args.ground_truth,
        args.output,
        args.class_mapping,
    )
    print(f"Ornek: {n}  benzersiz ClassId: {u}")
    print(f"Cikti: {args.output.resolve()}")


if __name__ == "__main__":
    main()
