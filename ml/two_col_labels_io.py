"""Iki sutunlu (kok, ClassId) ve ClassId/TR haritasi CSV okuyuculari."""

from __future__ import annotations

import csv
import re
from pathlib import Path


def normalize_stem(cell: str) -> str:
    s = (cell or "").strip().strip('"')
    s = re.sub(r"\.mp4$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_(color|depth)$", "", s, flags=re.IGNORECASE)
    return s


def load_instances(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        for row in reader:
            if len(row) < 2:
                continue
            stem = normalize_stem(row[0])
            if not stem:
                continue
            try:
                cid = int(float(row[1]))
            except ValueError:
                continue
            mapping[stem] = cid
    return mapping


def load_class_mapping(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("class_mapping bos veya okunamadi.")
        fields = {n.strip() for n in reader.fieldnames}
        if "ClassId" not in fields or "TR" not in fields:
            raise SystemExit(f"class_mapping'da ClassId ve TR olmali. Bulunan: {reader.fieldnames}")
        out: dict[int, str] = {}
        for r in reader:
            try:
                cid = int(float(r.get("ClassId", "").strip()))
            except (ValueError, TypeError, AttributeError):
                continue
            tr = (r.get("TR") or "").strip()
            if tr:
                out[cid] = tr
        return out
