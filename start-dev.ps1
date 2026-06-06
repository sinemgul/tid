# Proje kökünden çalıştırın: .\start-dev.ps1
# İki ayrı pencerede backend (8000) ve frontend (5173) başlatır.

$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

Write-Host "Backend: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Frontend: http://127.0.0.1:5173" -ForegroundColor Green

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location -LiteralPath '$backend'; uvicorn main:app --host 127.0.0.1 --port 8000"
)

Start-Sleep -Seconds 1

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location -LiteralPath '$frontend'; npm run dev"
)

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5173/"
