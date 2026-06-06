"""
ChaLearn:
- validation_labels.zip (ayri sifre)
- test_labels.zip (ayri sifre)
- istege bagli: test_set_xsaft57.zip.001 + parcalar (genelde test verisi ile ayni sifre)

Cikti: data/chalearn_unpacked/, etiketlerim_chalearn_val.csv, etiketlerim_chalearn_test.csv

Ortam degiskenleri (onerilir; sifreleri repoya yazmayin):
  CHALEARN_PWD_VALIDATION_LABELS
  CHALEARN_PWD_TEST_LABELS
  CHALEARN_PWD_TEST_VIDEO          (yoksa TEST_LABELS ile ayni kabul edilir)

Tek sifre kullanan paketler icin: CHALEARN_ZIP_PASSWORD (hepsi icin fallback)

Kullanim:
  cd ml
  set CHALEARN_PWD_VALIDATION_LABELS=...
  set CHALEARN_PWD_TEST_LABELS=...
  set CHALEARN_PWD_TEST_VIDEO=...
  python chalearn_prepare.py --extract_test_video
"""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

from import_train_labels import write_etiketlerim_from_labels_csv


def find_source_dir(guess: Path | None) -> Path:
    if guess and guess.is_dir():
        return guess
    import glob

    cands = glob.glob(r"c:/Users/sinem/Desktop/Yeni klas* (2)")
    if not cands:
        raise SystemExit("Kaynak klasor bulunamadi. --source_dir verin.")
    return Path(cands[0])


def extract_zip(zip_path: Path, dest: Path, passwords: list[str]) -> Path:
    last_err: Exception | None = None
    tried: list[str] = []
    for pw in passwords:
        if not pw or pw in tried:
            continue
        tried.append(pw)
        if dest.exists():
            for p in sorted(dest.glob("**/*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest, pwd=pw.encode("utf-8"))
                if not z.namelist():
                    raise SystemExit(f"Bos arsiv: {zip_path}")
            for root, _, files in os.walk(dest):
                if "ground_truth.csv" in files:
                    return Path(root) / "ground_truth.csv"
            raise SystemExit(f"ground_truth.csv bulunamadi: {dest}")
        except (RuntimeError, zipfile.BadZipFile) as e:
            last_err = e
            continue
    raise SystemExit(f"Zip acilamadi: {zip_path}  Son hata: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=None)
    parser.add_argument("--password", type=str, default=None, help="Tek sifre (tum zip + video)")
    parser.add_argument("--password_validation", type=str, default=None)
    parser.add_argument("--password_test_labels", type=str, default=None)
    parser.add_argument("--password_test_video", type=str, default=None)
    parser.add_argument("--out_root", type=Path, default=Path("data/chalearn_unpacked"))
    parser.add_argument("--class_mapping", type=Path, default=None)
    parser.add_argument(
        "--extract_test_video",
        action="store_true",
        help="test_set_xsaft57.zip.001 WinRAR ile cikar (WinRAR kurulu olmali)",
    )
    parser.add_argument(
        "--video_out",
        type=Path,
        default=Path("../_data_test"),
        help="Test video cikis klasoru (repo kokune gore)",
    )
    args = parser.parse_args()

    fallback = args.password or os.environ.get("CHALEARN_ZIP_PASSWORD", "")
    pv = args.password_validation or os.environ.get("CHALEARN_PWD_VALIDATION_LABELS", fallback)
    pt = args.password_test_labels or os.environ.get("CHALEARN_PWD_TEST_LABELS", fallback)
    px = args.password_test_video or os.environ.get("CHALEARN_PWD_TEST_VIDEO", pt)

    if not pv or not pt:
        raise SystemExit(
            "Validation ve test etiket sifreleri gerekli. "
            "CHALEARN_PWD_VALIDATION_LABELS ve CHALEARN_PWD_TEST_LABELS veya --password (tek)."
        )

    src = find_source_dir(args.source_dir)
    val_zip = src / "validation_labels.zip"
    test_zip = src / "test_labels.zip"
    if not val_zip.is_file():
        raise SystemExit(f"Yok: {val_zip}")
    if not test_zip.is_file():
        raise SystemExit(f"Yok: {test_zip}")

    val_dir = args.out_root / "validation"
    test_dir = args.out_root / "test"
    gt_val = extract_zip(val_zip, val_dir, [pv, fallback] if pv != fallback else [pv])
    gt_test = extract_zip(test_zip, test_dir, [pt, fallback] if pt != fallback else [pt])

    out_val = Path("etiketlerim_chalearn_val.csv")
    out_test = Path("etiketlerim_chalearn_test.csv")

    n1, u1 = write_etiketlerim_from_labels_csv(gt_val, out_val, args.class_mapping)
    n2, u2 = write_etiketlerim_from_labels_csv(gt_test, out_test, args.class_mapping)

    print(f"Val  : {gt_val} -> {out_val.resolve()}  ornek={n1} sinif={u1}")
    print(f"Test : {gt_test} -> {out_test.resolve()} ornek={n2} sinif={u2}")

    if args.extract_test_video:
        part1 = src / "test_set_xsaft57.zip.001"
        if not part1.is_file():
            print("--extract_test_video: test_set_xsaft57.zip.001 bulunamadi, atlandı.")
        else:
            args.video_out.mkdir(parents=True, exist_ok=True)
            rar = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "WinRAR" / "WinRAR.exe"
            if not rar.is_file():
                raise SystemExit(f"WinRAR yok: {rar}")
            for pwd_try in [px, pt, fallback]:
                if not pwd_try:
                    continue
                r = subprocess.run(
                    [str(rar), "x", f"-p{pwd_try}", "-y", str(part1), str(args.video_out) + "/"],
                    cwd=str(src),
                    timeout=60 * 60,
                )
                if r.returncode == 0:
                    print(f"Test videolari: {args.video_out.resolve()} (pwd ok)")
                    break
                print(f"WinRAR exit {r.returncode}, baska sifre deneniyor...")
            else:
                print("Test video arsivi acilamadi; WinRAR ve sifreyi elle deneyin.")

    print("\nDegerlendirme ornegi:")
    print(
        "  python eval_checkpoint.py --labels etiketlerim_chalearn_test.csv "
        '--video_dir "..\\\\_data_test\\\\test" --checkpoint output\\\\sign_r3d\\\\best.pt'
    )


if __name__ == "__main__":
    main()
