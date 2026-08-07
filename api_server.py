from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta

from src.inference import CropHealthPredictor
from src.data.openmeteo_fetcher import OpenMeteoAgriFetcher

app = FastAPI(
    title="Uydu Verileri ile Rekolte ve Kuraklık Stresi Takip Sistemi API",
    description="Sentinel-2 ve Meteoroloji Zaman Serisi Yapay Zeka Servisi",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None

@app.on_event("startup")
def load_ai_models():
    global predictor
    try:
        predictor = CropHealthPredictor()
        print("AI Models successfully loaded into API Server.")
    except Exception as e:
        print(f"Warning loading AI models: {e}")

class FieldCoordinateRequest(BaseModel):
    latitude: float
    longitude: float
    field_name: Optional[str] = "Tarla #1"
    is_irrigated: Optional[int] = 1
    soil_type: Optional[int] = 2

@app.get("/")
def root():
    return {
        "system": "Uydu Verileri ile Rekolte ve Kuraklık Stresi Takip Sistemi",
        "status": "online",
        "version": "2.0.0",
        "ai_models_loaded": predictor is not None,
        "endpoints": {
            "predict_coordinates": "/api/predict-coordinates",
            "health": "/health"
        }
    }

@app.post("/api/predict-coordinates")
def predict_field_by_coordinates(req: FieldCoordinateRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="AI models not loaded.")
        
    fetcher = OpenMeteoAgriFetcher()
    end_dt = datetime.now().strftime("%Y-%m-%d")
    start_dt = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    df_live = fetcher.fetch_field_climate_history(lat=req.latitude, lon=req.longitude, start_date=start_dt, end_date=end_dt)
    if df_live is None or df_live.empty:
        raise HTTPException(status_code=400, detail="Could not fetch climate data for target coordinates.")
        
    df_sentinel_weather = fetcher.aggregate_to_sentinel_cadence(df_live, interval_days=5)
    
    # Synthesize NDVI/NDWI curves for the live field trajectory
    steps = len(df_sentinel_weather)
    base_ndvi = 0.20 + 0.65 * (df_sentinel_weather["soil_moisture_%"] / 40.0)
    base_ndwi = (df_sentinel_weather["soil_m_deep"] - 0.15) * 1.5
    
    df_sentinel_weather["ndvi"] = base_ndvi.clip(0.08, 0.90)
    df_sentinel_weather["ndwi"] = base_ndwi.clip(-0.35, 0.50)
    df_sentinel_weather["temp_c"] = df_sentinel_weather["temp_mean_c"]
    df_sentinel_weather["soil_moisture"] = df_sentinel_weather["soil_moisture_%"]
    df_sentinel_weather["soil_type"] = req.soil_type
    df_sentinel_weather["is_irrigated"] = req.is_irrigated
    
    res = predictor.predict_field_health(df_sentinel_weather)
    res["field_name"] = req.field_name
    res["latitude"] = req.latitude
    res["longitude"] = req.longitude
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
