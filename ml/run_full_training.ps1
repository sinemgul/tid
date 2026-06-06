# Tam egitim (val setinin tamami) + galeri. Train arsivi yoksa val kullanilir.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== 1/2 Frame model egitimi (tum val, 12 epoch) ===" -ForegroundColor Cyan
$env:PYTHONUNBUFFERED = "1"
python train_sign_frame.py --epochs 12 --max_samples 0 --batch_size 12 --lr 1e-4

Write-Host "=== 2/2 Galeri (tum val videolari) ===" -ForegroundColor Cyan
python build_gallery.py --max_items 0

Write-Host "Bitti. Backend'i yeniden baslatin (uvicorn)." -ForegroundColor Green
