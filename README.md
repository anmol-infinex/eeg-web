# Seizure-Sentry

Real-time 3-class EEG classification web platform (Normal / Pre-Seizure / Seizure).

> **Research prototype. Not a medical device.**

## Architecture

```
OpenBCI Cyton+Daisy → USB dongle → Browser (Web Serial)
                                 → WebSocket → FastAPI backend → ML Model → Dashboard
```

- **Backend**: FastAPI + Keras 3 (PyTorch CPU) on **Render** (Docker)
- **Frontend**: React 18 + TypeScript + Vite + Tailwind on **Vercel**
- **Model**: `tusz_3class_eeg_model.h5` — input (None, 1280, 22) float32, 256 Hz

## Quick start

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 10000
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local: set VITE_API_URL=http://localhost:10000
npm run dev
```

## Deployment

### Backend (Render)
1. Push to GitHub
2. Connect repo to Render using `render.yaml` blueprint
3. Set env vars in Render dashboard: `ACCESS_KEY`, `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`
4. Backend URL: `https://seizure-sentry-api.onrender.com`

### Frontend (Vercel)
1. Import GitHub repo in Vercel
2. Root directory: `frontend`
3. Set env var: `VITE_API_URL=https://seizure-sentry-api.onrender.com`
4. Deploy

## Project structure

```
seizure-sentry/
  backend/
    app/
      main.py          FastAPI app + endpoints
      ml_runner.py     Keras3/PyTorch ModelRunner
      synthetic.py     Synthetic EEG generator
      adapter.py       Channel remapping + resampling
      config.py        Environment settings
    models/
      tusz_3class_eeg_model.h5
    Dockerfile
    requirements.txt
  frontend/
    src/
      pages/           Overview, LiveMonitor, Devices, Recordings, Alerts, ModelCard, Settings
      components/      EegCanvas, PredictionTimeline, StateChip, StatusChip, Banner, Button, etc.
      lib/             api.ts, wsClient.ts
    index.html
    vite.config.ts
    tailwind.config.ts
  tests/e2e/           366 automated tests (Tiers 1-4)
  render.yaml          Render deployment blueprint
  vercel.json          Vercel deployment config
  .github/workflows/   CI pipeline
```
