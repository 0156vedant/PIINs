"""
Mock FastAPI Backend for Physi-Cast
Lightweight demo server without PyTorch dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import json
from datetime import datetime

# Request/Response Models
class CoordinatePoint(BaseModel):
    x: float
    y: float
    z: float
    t: float

class LocationQuery(BaseModel):
    latitude: float
    longitude: float
    altitude: Optional[float] = 0
    forecast_hours: Optional[int] = 24

class PredictionResponse(BaseModel):
    timestamp: str
    location: Dict[str, float]
    wind_u: float
    wind_v: float
    wind_w: float
    pressure: float
    temperature: float
    confidence: float

# FastAPI Application
app = FastAPI(
    title="Physi-Cast API (Demo)",
    description="Physics-Informed Neural Network Climate Forecasting",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": "cpu",
        "timestamp": datetime.utcnow().isoformat()
    }

# Model info
@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "Physics-Informed Neural Network (PINN)",
        "input_features": 4,
        "output_features": 5,
        "physics_constraints": ["Navier-Stokes", "Thermal Diffusion", "Continuity"],
        "device": "cpu",
        "status": "ready"
    }

# Point prediction
@app.post("/predict/point", response_model=PredictionResponse)
async def predict_point(point: CoordinatePoint):
    """Predict weather at a single coordinate point"""
    
    # Generate realistic demo predictions
    np.random.seed(int(point.x + point.y + point.z + point.t) % 10000)
    
    # Temperature based on time of day and location
    base_temp = 283.15 + 10 * np.sin(2 * np.pi * (point.t % 86400) / 86400)
    temp = base_temp + np.random.normal(0, 1)
    
    # Wind components
    wind_u = 3 + 2 * np.sin(point.t / 3600) + np.random.normal(0, 0.5)
    wind_v = 2 + 1.5 * np.cos(point.t / 3600) + np.random.normal(0, 0.5)
    wind_w = 0.1 + np.random.normal(0, 0.05)
    
    # Pressure
    pressure = 101325 + 100 * np.sin(point.t / 7200) + np.random.normal(0, 50)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "location": {
            "latitude": point.y,
            "longitude": point.x,
            "altitude": point.z
        },
        "wind_u": float(wind_u),
        "wind_v": float(wind_v),
        "wind_w": float(wind_w),
        "pressure": float(pressure),
        "temperature": float(temp),
        "confidence": 0.75 + np.random.uniform(-0.05, 0.2)
    }

# Grid predictions
@app.post("/predict/grid")
async def predict_grid(query: LocationQuery):
    """Get grid predictions"""
    grid_size = 10
    predictions = {
        "temperature": [],
        "wind_u": [],
        "wind_v": [],
        "pressure": []
    }
    
    for i in range(grid_size):
        for j in range(grid_size):
            predictions["temperature"].append(283.15 + np.random.normal(0, 3))
            predictions["wind_u"].append(3 + np.random.normal(0, 1))
            predictions["wind_v"].append(2 + np.random.normal(0, 1))
            predictions["pressure"].append(101325 + np.random.normal(0, 100))
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "grid_shape": (grid_size, grid_size),
        "data": predictions,
        "metadata": {
            "center_lat": query.latitude,
            "center_lon": query.longitude,
            "forecast_hours": query.forecast_hours
        }
    }

# Weather alerts
@app.post("/alerts/extreme-weather")
async def get_weather_alerts(query: LocationQuery):
    """Get extreme weather alerts"""
    
    # Randomly decide whether to return alerts for demo
    has_alerts = np.random.random() < 0.3
    
    if has_alerts:
        return {
            "alerts": [
                {
                    "alert_type": "High Wind Warning",
                    "severity": "HIGH",
                    "description": "Wind speeds expected to reach 40 km/h with gusts to 60 km/h.",
                    "confidence": 0.85,
                    "recommended_actions": [
                        "Secure outdoor items",
                        "Avoid outdoor activities",
                        "Monitor weather updates"
                    ]
                }
            ]
        }
    else:
        return {"alerts": []}

# Stats endpoint
@app.get("/stats/last-prediction")
async def get_last_stats():
    """Get stats from last prediction"""
    return {
        "last_prediction_time": datetime.utcnow().isoformat(),
        "predictions_today": 42,
        "average_confidence": 0.78,
        "locations_covered": 156
    }

if __name__ == "__main__":
    import uvicorn
    print("[STARTUP] Mock API Server starting...")
    print("[STARTUP] Listening on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
