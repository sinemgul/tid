"""
Etiket CSV -> (video_koku -> sinif id) ve id2name haritalari.

Oncelik: herhangi bir satirda sinif_id doluysa tum egitim satirlari icin sayisal sinif_id kullanilir.
Aksi halde sadece tr_isaret_veya_cumle dolu satirlar siniflenir (0..K-1).
Her iki sutun da bos satirlar egitim disi birakilir.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabelMaps:
    stem_to_id: dict[str, int]
    id_to_name: dict[str, str]

    def num_classes(self) -> int:
        if not self.stem_to_id:
            return 0
        return max(self.stem_to_id.values()) + 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stem_to_id": self.stem_to_id,
            "id_to_name": self.id_to_name,
            "num_classes": self.num_classes(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "LabelMaps":
        data = json.loads(path.read_text(encoding="utf-8"))
        return LabelMaps(stem_to_id=data["stem_to_id"], id_to_name=data["id_to_name"])


def build_from_csv(labels_csv: Path) -> LabelMaps:
    rows: list[dict[str, str]] = []
    with labels_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_koku", "tr_isaret_veya_cumle", "sinif_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"CSV basliklari eksik. Gerekli: {required}. Var: {reader.fieldnames}")
        for r in reader:
            rows.append({k: (r.get(k) or "").strip() for k in reader.fieldnames})

    use_numeric = any(r["sinif_id"] for r in rows)
    stem_to_id: dict[str, int] = {}
    id_to_name: dict[str, str] = {}

    if use_numeric:
        for r in rows:
            stem = r["video_koku"]
            if not stem or not r["sinif_id"]:
                continue
            y = int(r["sinif_id"])
            name = r["tr_isaret_veya_cumle"] or f"sinif_{y}"
            stem_to_id[stem] = y
            id_to_name[str(y)] = name
        return LabelMaps(stem_to_id=stem_to_id, id_to_name=id_to_name)

    text_to_id: dict[str, int] = {}
    next_id = 0
    for r in rows:
        stem = r["video_koku"]
        t = r["tr_isaret_veya_cumle"]
        if not stem or not t:
            continue
        if t not in text_to_id:
            text_to_id[t] = next_id
            id_to_name[str(next_id)] = t
            next_id += 1
        stem_to_id[stem] = text_to_id[t]

    return LabelMaps(stem_to_id=stem_to_id, id_to_name=id_to_name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Etiket CSV'den JSON harita uret.")
    parser.add_argument("--labels_csv", type=Path, default=Path("etiketlerim_chalearn_val.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/label_map.json"))
    args = parser.parse_args()
    m = build_from_csv(args.labels_csv)
    m.save(args.output)
    print(f"Ornek sayisi: {len(m.stem_to_id)} sinif sayisi (C): {m.num_classes()}")
    print(f"Yazildi: {args.output.resolve()}")


if __name__ == "__main__":
    main()
