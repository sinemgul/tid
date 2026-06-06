# Isaret Dili Web Demo

Web arayuzu ve FastAPI backend (mock / sonraki entegrasyon).

## Proje yapisi

```text
root/
  frontend/  -> React + Vite + Tailwind + Zustand
  backend/   -> FastAPI
  ml/        -> ChaLearn etiket hazirligi, istege bagli egitim scriptleri
  README.md
```

## Veri ozeti

- **Ana akis:** `ml/chalearn_prepare.py` → `etiketlerim_chalearn_val.csv`,
  `etiketlerim_chalearn_test.csv` (+ istege bagli test videolari). Ayrinti:
  **ml/OKU_BENI.txt**.
- **Train etiket:** `ml/import_train_labels.py` → `etiketlerim_train.csv`
  (goruntu arsivi bu repoda kullanilmiyor).
- Isaret videosu: **ml/train_sign_video.py**, **ml/predict_sign_video.py**.
- Backend ozeti: **GET /datasets/chalearn**.
- Rapor notlari: **ml/RAPOR_ICIN_OZET.txt**.

## Backend kurulum

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API: `http://127.0.0.1:8000`

- `GET /health`
- `POST /predict`
- `POST /llm-correct`

## Frontend kurulum

```bash
cd frontend
copy .env.example .env
npm install
npm run dev
```

Adres: `http://127.0.0.1:5173`

## Ozellikler

- Dark mode arayuz, landing (`/`) ve uygulama (`/app`)
- Webcam ve periyodik `predict` cagrisi
- Zustand state (`isStreaming`, `rawText`, `correctedText`)

## On egitilmis model (repoda)

Repoda `ml/output/sign_frame/best.pt` ve `ml/artifacts/gallery.npz` vardir.
Clone sonrasi ek egitim gerekmez; backend baslatinca gercek tahmin calisir.

**Videolar repoda degil** (`_data_val`, `_data_test`). Kendi AUTSL videolarinizla
yeniden egitmek icin `ml/run_full_training.ps1` kullanin.
