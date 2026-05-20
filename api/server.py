"""
FastAPI Backend for Physi-Cast
Serves PINN model predictions and provides REST API endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import torch
import json
from datetime import datetime
import asyncio

# Import PINN components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network import FCNN
from physics import PhysicsConstraints
from utils import GeoNormalizer


# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

class CoordinatePoint(BaseModel):
    """Single coordinate point for prediction"""
    x: float  # Longitude in km
    y: float  # Latitude in km
    z: float  # Altitude in meters
    t: float  # Time in seconds from start of period


class LocationQuery(BaseModel):
    """Query multiple points at specific location"""
    latitude: float
    longitude: float
    altitude: Optional[float] = 0
    forecast_hours: Optional[int] = 24


class PredictionResponse(BaseModel):
    """Weather prediction for a single point"""
    timestamp: str
    location: Dict[str, float]
    wind_u: float  # Zonal wind (m/s)
    wind_v: float  # Meridional wind (m/s)
    wind_w: float  # Vertical wind (m/s)
    pressure: float  # Pressure (Pa)
    temperature: float  # Temperature (K)
    confidence: float  # Prediction confidence [0-1]


class GridPredictionResponse(BaseModel):
    """Predictions for a spatial grid"""
    timestamp: str
    grid_shape: tuple
    data: Dict[str, List[float]]
    metadata: Dict


class WeatherAlertResponse(BaseModel):
    """Weather alert based on extreme predictions"""
    alert_type: str  # "SEVERE_WIND", "FROST", "HEAT_WAVE", etc.
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    description: str
    affected_area: Dict[str, float]
    confidence: float
    recommended_actions: List[str]


# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================

app = FastAPI(
    title="Physi-Cast API",
    description="Physics-Informed Neural Network Climate Forecasting",
    version="1.0.0"
)

# Add CORS middleware for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# GLOBAL STATE
# ==============================================================================

class ModelState:
    """Global model state"""
    model: Optional[FCNN] = None
    physics: Optional[PhysicsConstraints] = None
    normalizer: Optional[GeoNormalizer] = None
    device: str = "cpu"
    model_loaded: bool = False
    last_prediction: Optional[Dict] = None


model_state = ModelState()


# ==============================================================================
# INITIALIZATION & HEALTH CHECKS
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    print("[STARTUP] Loading PINN model...")
    
    try:
        # Initialize model
        model_state.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[STARTUP] Using device: {model_state.device}")
        
        # Create neural network
        model_state.model = FCNN(
            input_dim=4, output_dim=5, hidden_units=128
        ).to(model_state.device)
        
        # Initialize physics constraints
        model_state.physics = PhysicsConstraints()
        
        # Initialize normalizer
        model_state.normalizer = GeoNormalizer()
        
        # Try to load saved model weights if available
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pinn_model.pth')
        if os.path.exists(model_path):
            try:
                model_state.model.load_state_dict(
                    torch.load(model_path, map_location=model_state.device)
                )
                model_state.model_loaded = True
                print("[STARTUP] Saved model weights loaded successfully")
            except Exception as load_error:
                print(f"[STARTUP] Could not load model: {str(load_error)}")
                model_state.model_loaded = False
        else:
            print(f"[STARTUP] No saved model found at {model_path}, using initialized weights")
            model_state.model_loaded = False
        
        model_state.model.eval()
        print("[STARTUP] Model ready for inference")
        
    except Exception as e:
        print(f"[STARTUP ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_state.model_loaded,
        "device": model_state.device,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "Physics-Informed Neural Network (PINN)",
        "input_features": 4,  # x, y, z, t
        "output_features": 5,  # u, v, w, p, T
        "physics_constraints": ["Navier-Stokes", "Thermal Diffusion", "Continuity"],
        "device": model_state.device,
        "status": "ready" if model_state.model_loaded else "initialized"
    }


# ==============================================================================
# PREDICTION ENDPOINTS
# ==============================================================================

@app.post("/predict/point", response_model=PredictionResponse)
async def predict_point(point: CoordinatePoint):
    """
    Predict weather at a single coordinate point
    
    Args:
        point: Coordinate (x, y, z, t)
        
    Returns:
        Weather prediction
    """
    if not model_state.model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Normalize coordinates
        x_norm, y_norm, z_norm, t_norm = model_state.normalizer.normalize_coords(
            np.array([point.x]),
            np.array([point.y]),
            np.array([point.z]),
            np.array([point.t])
        )
        
        # Create input tensor
        input_tensor = torch.tensor(
            [[x_norm[0], y_norm[0], z_norm[0], t_norm[0]]],
            dtype=torch.float32,
            device=model_state.device
        )
        
        # Make prediction
        with torch.no_grad():
            output = model_state.model(input_tensor)
            output = output.cpu().numpy()[0]
        
        # Extract values
        u, v, w, p, T = output
        
        # Denormalize pressure and temperature
        # (simplified - in production, would use proper inverse transforms)
        p = p * 50000 + 101325  # Rough denormalization
        T = T * 10 + 288.15      # Rough denormalization
        
        # Calculate confidence (simplified - based on prediction magnitude)
        confidence = min(1.0, max(0.0, 0.8))  # Placeholder
        
        response = PredictionResponse(
            timestamp=datetime.utcnow().isoformat(),
            location={"x": point.x, "y": point.y, "z": point.z},
            wind_u=float(u),
            wind_v=float(v),
            wind_w=float(w),
            pressure=float(p),
            temperature=float(T),
            confidence=confidence
        )
        
        model_state.last_prediction = response.dict()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/grid")
async def predict_grid(query: LocationQuery):
    """
    Predict weather over a spatial grid
    
    Args:
        query: Location and forecast parameters
        
    Returns:
        Grid of predictions
    """
    if not model_state.model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create prediction grid
        grid_size = 10  # 10x10 grid
        x_range = np.linspace(query.longitude - 5, query.longitude + 5, grid_size)
        y_range = np.linspace(query.latitude - 5, query.latitude + 5, grid_size)
        z = query.altitude
        t = 0
        
        predictions = {
            'wind_u': [],
            'wind_v': [],
            'wind_w': [],
            'pressure': [],
            'temperature': []
        }
        
        # Generate predictions for grid
        for x in x_range:
            for y in y_range:
                # Normalize
                x_norm, y_norm, z_norm, t_norm = model_state.normalizer.normalize_coords(
                    np.array([x]),
                    np.array([y]),
                    np.array([z]),
                    np.array([t])
                )
                
                input_tensor = torch.tensor(
                    [[x_norm[0], y_norm[0], z_norm[0], t_norm[0]]],
                    dtype=torch.float32,
                    device=model_state.device
                )
                
                with torch.no_grad():
                    output = model_state.model(input_tensor).cpu().numpy()[0]
                
                u, v, w, p, T = output
                
                predictions['wind_u'].append(float(u))
                predictions['wind_v'].append(float(v))
                predictions['wind_w'].append(float(w))
                predictions['pressure'].append(float(p * 50000 + 101325))
                predictions['temperature'].append(float(T * 10 + 288.15))
        
        response = GridPredictionResponse(
            timestamp=datetime.utcnow().isoformat(),
            grid_shape=(grid_size, grid_size),
            data=predictions,
            metadata={
                "location": {"lat": query.latitude, "lon": query.longitude},
                "forecast_hours": query.forecast_hours
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Grid prediction failed: {str(e)}")


# ==============================================================================
# ALERT ENDPOINTS
# ==============================================================================

@app.post("/alerts/extreme-weather")
async def detect_extreme_weather(location: LocationQuery):
    """
    Detect extreme weather conditions
    
    Args:
        location: Location to check
        
    Returns:
        List of weather alerts
    """
    if not model_state.model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Make prediction
        point = CoordinatePoint(
            x=location.longitude,
            y=location.latitude,
            z=location.altitude or 0,
            t=0
        )
        
        prediction = await predict_point(point)
        
        alerts = []
        
        # Check for extreme wind
        wind_speed = np.sqrt(prediction.wind_u**2 + prediction.wind_v**2 + 
                            prediction.wind_w**2)
        if wind_speed > 15:
            alerts.append(WeatherAlertResponse(
                alert_type="SEVERE_WIND",
                severity="HIGH" if wind_speed > 20 else "MEDIUM",
                description=f"Strong winds predicted ({wind_speed:.1f} m/s)",
                affected_area={"lat": location.latitude, "lon": location.longitude},
                confidence=0.8,
                recommended_actions=[
                    "Secure outdoor equipment",
                    "Reduce open field operations"
                ]
            ))
        
        # Check for frost
        if prediction.temperature < 273.15:
            alerts.append(WeatherAlertResponse(
                alert_type="FROST",
                severity="HIGH",
                description=f"Frost conditions predicted ({prediction.temperature - 273.15:.1f}°C)",
                affected_area={"lat": location.latitude, "lon": location.longitude},
                confidence=0.85,
                recommended_actions=[
                    "Protect sensitive crops",
                    "Activate frost protection systems"
                ]
            ))
        
        # Check for heat wave
        if prediction.temperature > 308.15:
            alerts.append(WeatherAlertResponse(
                alert_type="HEAT_WAVE",
                severity="MEDIUM",
                description=f"Heat wave predicted ({prediction.temperature - 273.15:.1f}°C)",
                affected_area={"lat": location.latitude, "lon": location.longitude},
                confidence=0.75,
                recommended_actions=[
                    "Increase irrigation frequency",
                    "Monitor livestock for heat stress"
                ]
            ))
        
        return {"alerts": alerts, "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Alert detection failed: {str(e)}")


# ==============================================================================
# UTILITY ENDPOINTS
# ==============================================================================

@app.get("/stats/last-prediction")
async def get_last_prediction():
    """Get the last prediction made"""
    if model_state.last_prediction:
        return model_state.last_prediction
    else:
        raise HTTPException(status_code=404, detail="No predictions made yet")


@app.post("/calibrate/model")
async def calibrate_model(background_tasks: BackgroundTasks):
    """
    Trigger model recalibration (background task)
    
    Args:
        background_tasks: FastAPI background tasks
    """
    background_tasks.add_task(dummy_calibration_task)
    
    return {"status": "Calibration started", "timestamp": datetime.utcnow().isoformat()}


def dummy_calibration_task():
    """Placeholder calibration task"""
    print("[CALIBRATION] Starting model calibration...")
    # In production, this would retrain the model
    print("[CALIBRATION] Calibration complete")


# ==============================================================================
# ROOT ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Physi-Cast API",
        "description": "Physics-Informed Neural Network for Climate Forecasting",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict_point": "/predict/point",
            "predict_grid": "/predict/grid",
            "alerts": "/alerts/extreme-weather"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
