"""
AUTSL metin korpusu üretimi.

Önemli: val_set zip içindeki .mp4 dosyaları doğrudan dil modeli eğitim verisi değildir.
Eğitim/DM için Türkçe metin satırları gerekir. AUTSL paketindeki CSV'lerle
video isimleri ClassId'ye, ClassId de Türkçe gloss (TR) sütununa bağlanır.

Beklenen dosyalar (örnek isimler resmi dağıtımınıza göre değişebilir):
- split dosyası: 2 sütun, başlıksız — örn. signer11_sample100, 42
- sınıf haritası: ClassId, TR

Referans: OpenHands AUTSL okuyucusu (split + ClassId -> TR eşlemesi).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="AUTSL CSV'lerinden Türkçe gloss metin dosyası üret.")
    parser.add_argument(
        "--split_csv",
        type=Path,
        required=True,
        help="İki sütunlu, başlıksız split CSV (video kök adı, ClassId).",
    )
    parser.add_argument(
        "--class_mapping_csv",
        type=Path,
        required=True,
        help="ClassId ve TR sütunları içeren sınıf eşlemesi (resmi AUTSL).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corpus_autsl_gloss.txt"),
        help="Çıktı: her satırda bir Türkçe gloss veya birleştirilmiş satırlar.",
    )
    parser.add_argument(
        "--mode",
        choices=("lines", "paragraph"),
        default="lines",
        help="lines: her örnek ayrı satır | paragraph: ardışık N gloss tek satırda.",
    )
    parser.add_argument("--paragraph_size", type=int, default=8, help="paragraph modunda kaç gloss birleşsin.")
    args = parser.parse_args()

    split_df = pd.read_csv(args.split_csv, header=None)
    if split_df.shape[1] < 2:
        raise ValueError("split_csv en az iki sütun içermeli.")

    map_df = pd.read_csv(args.class_mapping_csv)
    if "ClassId" not in map_df.columns or "TR" not in map_df.columns:
        raise ValueError("class_mapping_csv içinde ClassId ve TR sütunları olmalı.")

    id_to_tr = dict(zip(map_df["ClassId"], map_df["TR"]))
    glosses: list[str] = []
    for _, row in split_df.iterrows():
        cid = row[1]
        if pd.isna(cid):
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        tr = id_to_tr.get(cid_int)
        if tr is None:
            continue
        glosses.append(str(tr).strip())

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "lines":
        args.output.write_text("\n".join(glosses) + "\n", encoding="utf-8")
    else:
        chunks: list[str] = []
        buf: list[str] = []
        for g in glosses:
            buf.append(g)
            if len(buf) >= args.paragraph_size:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))
        args.output.write_text("\n".join(chunks) + "\n", encoding="utf-8")

    print(f"Yazildi: {args.output.resolve()} satir: {len(glosses)} gloss, mod: {args.mode}")


if __name__ == "__main__":
    main()
