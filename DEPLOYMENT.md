# HINTSight Deployment

## Backend on Render

1. Push this repository to GitHub, including these runtime files:
   - `cnn4_tcn_best.pt`
   - `hintsight_fusion_model.pt`
   - `iris_model (1).keras`
   - `face_landmarker.task`
2. In Render, choose **New > Blueprint** and select the repository.
3. Render reads `render.yaml`, builds the `Dockerfile`, and starts FastAPI.
4. Set `GOOGLE_MAPS_API_KEY` in the Render service environment if care-finder search is needed.
5. Check the deployment at:

```text
https://YOUR_SERVICE.onrender.com/health
```

The health response should contain `"pipeline_loaded": true`.

## Local Docker Test

From the repository root:

```powershell
docker build -t hintsight-api .
docker run --rm -p 8000:8000 hintsight-api
```

Then open `http://localhost:8000/health`.

## Mobile App

Set the deployed API URL before starting or building Expo:

```powershell
cd mobile
$env:EXPO_PUBLIC_API_BASE_URL = "https://YOUR_SERVICE.onrender.com"
npx.cmd expo start
```

For a development phone test, use the laptop Wi-Fi address instead:

```powershell
$env:EXPO_PUBLIC_API_BASE_URL = "http://YOUR_LAPTOP_WIFI_IP:8000"
npx.cmd expo start
```

The phone and laptop must be on the same Wi-Fi for the local URL. The app sends the ten metadata fields and the recorded video to `POST /predict`.

## Important Runtime Notes

- The backend requires Python 3.12 because TensorFlow loads `iris_model (1).keras`.
- The late-fusion checkpoint must be retrained with `meta_dim=10`.
- This is a screening prototype, not a diagnostic device.
- Do not store uploaded patient videos permanently unless privacy, consent, and retention controls are implemented.
