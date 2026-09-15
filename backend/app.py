"""
FastAPI backend that exposes the HINTSight fused screening pipeline
(vHIT video -> CNN branch + clinical quiz -> metadata branch -> late-fusion MLP)
to the mobile app over HTTP.

Run with:
    .venv\\Scripts\\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

import json
import math
import os
import tempfile
from typing import Literal, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set this in your shell before starting uvicorn, e.g.:
#   $env:GOOGLE_MAPS_API_KEY = "..."   (PowerShell)
# Needs the "Places API" enabled in Google Cloud Console.
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

FACILITY_SEARCH_PARAMS = {
    "er": {"type": "hospital", "keyword": "emergency room"},
    "neurologist": {"type": "doctor", "keyword": "neurologist"},
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, REPO_ROOT)

from phone_hit_fusion_pipeline import PhoneHITFusionPipeline  # noqa: E402

app = FastAPI(title="HINTSight Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[PhoneHITFusionPipeline] = None


@app.on_event("startup")
def load_pipeline():
    global _pipeline
    _pipeline = PhoneHITFusionPipeline(
        segmentation_model=os.path.join(REPO_ROOT, "iris_model (1).keras"),
        fusion_checkpoint=os.path.join(REPO_ROOT, "hintsight_fusion_model.pt"),
    )


class QuizAnswers(BaseModel):
    gender: str
    age: float
    hypertension: bool
    heart_disease: bool
    ever_married: bool
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": _pipeline is not None,
        "maps_configured": GOOGLE_MAPS_API_KEY is not None,
    }


def _haversine_meters(lat1, lng1, lat2, lng2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@app.get("/nearby-care")
def nearby_care(
    lat: float,
    lng: float,
    facility: Literal["er", "neurologist"] = "er",
    radius_m: int = 15000,
):
    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_MAPS_API_KEY is not set on the server. Set it and restart uvicorn.",
        )

    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "key": GOOGLE_MAPS_API_KEY,
        **FACILITY_SEARCH_PARAMS[facility],
    }

    response = requests.get(PLACES_NEARBY_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("status") not in ("OK", "ZERO_RESULTS"):
        raise HTTPException(status_code=502, detail=f"Places API error: {data.get('status')}")

    places = []
    for result in data.get("results", []):
        loc = result["geometry"]["location"]
        places.append(
            {
                "name": result.get("name"),
                "address": result.get("vicinity"),
                "lat": loc["lat"],
                "lng": loc["lng"],
                "place_id": result.get("place_id"),
                "rating": result.get("rating"),
                "open_now": result.get("opening_hours", {}).get("open_now"),
                "distance_m": round(_haversine_meters(lat, lng, loc["lat"], loc["lng"])),
                "maps_url": (
                    f"https://www.google.com/maps/search/?api=1"
                    f"&query={requests.utils.quote(result.get('name', ''))}"
                    f"&query_place_id={result.get('place_id')}"
                ),
            }
        )

    places.sort(key=lambda p: p["distance_m"])
    return {"facility": facility, "results": places[:10]}


@app.post("/predict")
async def predict(
    video: UploadFile = File(...),
    quiz: str = Form(...),  # JSON-encoded QuizAnswers
    side: str = Form("left"),
    duration_sec: float = Form(10.0),
):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    try:
        quiz_answers = QuizAnswers(**json.loads(quiz)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid quiz payload: {exc}")

    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        result = _pipeline.run(tmp_path, quiz_answers, duration_sec=duration_sec)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Processing failed: {exc}")
    finally:
        os.unlink(tmp_path)

    return result
