# HITSight

HITSight is a phone-based head impulse test screening prototype that combines eye segmentation, head and eye motion tracking, a HIT CNN, and a metadata late-fusion model.

## Repository Layout

- `AI_Training/`: training scripts, datasets, notebooks, model checkpoints, and training artifacts
- `Frontend/mobile/`: Expo mobile application
- `Frontend/static/`: static web assets
- `backend/`: FastAPI service
- `graphs/`: generated model-performance and factor-analysis figures
- Root Python files: shared inference, tracking, preprocessing, and fusion modules

## Run Locally

Start the backend with the Python 3.12 environment:

```powershell
.backend-venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

In a second terminal, start Expo:

```powershell
cd Frontend/mobile
$env:EXPO_PUBLIC_API_BASE_URL = "http://YOUR_LAPTOP_WIFI_IP:8000"
npx.cmd expo start --lan
```

The phone and laptop must share Wi-Fi. See `DEPLOYMENT.md` for Render and Docker instructions.

This is a screening research prototype, not a diagnostic device.
