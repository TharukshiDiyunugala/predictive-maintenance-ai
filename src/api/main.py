"""
FastAPI Application for Predictive Maintenance
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import MaintenancePredictor, AnomalyDetector
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="AI-driven predictive maintenance system API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api.cors_origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
predictor = MaintenancePredictor()
anomaly_detector = AnomalyDetector()


# Pydantic models
class SensorReading(BaseModel):
    """Single sensor reading"""
    sensor_id: str
    value: float
    unit: str
    timestamp: Optional[datetime] = None


class PredictionRequest(BaseModel):
    """Prediction request model"""
    equipment_id: str = Field(..., description="Equipment identifier")
    sensor_data: Dict[str, float] = Field(..., description="Sensor readings")
    timestamp: Optional[datetime] = None


class PredictionResponse(BaseModel):
    """Prediction response model"""
    equipment_id: str
    failure_prob: float
    failure_predicted: bool
    days_to_failure: Optional[int]
    severity: str
    confidence: float
    timestamp: str
    recommendations: List[str]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    equipment_id: str
    status: str
    failure_probability: float
    last_maintenance: Optional[str]
    next_maintenance: Optional[str]
    anomalies_detected: int


class DataIngestionRequest(BaseModel):
    """Data ingestion request"""
    equipment_id: str
    readings: List[SensorReading]


class AlertResponse(BaseModel):
    """Alert response model"""
    alert_id: str
    equipment_id: str
    severity: str
    message: str
    timestamp: str
    acknowledged: bool


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Predictive Maintenance API...")
    
    # Load failure prediction model
    try:
        model_path = config.get('models.failure_prediction.model_path')
        if Path(model_path).exists():
            predictor.load_model(model_path)
            logger.info("Failure prediction model loaded")
        else:
            logger.warning(f"Model not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    
    # Load anomaly detection model
    try:
        anomaly_path = config.get('models.anomaly_detection.model_path')
        if Path(anomaly_path).exists():
            anomaly_detector.load_model(anomaly_path)
            logger.info("Anomaly detection model loaded")
    except Exception as e:
        logger.error(f"Error loading anomaly detector: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Predictive Maintenance API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "predictor": predictor.model is not None,
            "anomaly_detector": anomaly_detector.model is not None
        }
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_failure(request: PredictionRequest):
    """
    Predict equipment failure probability
    """
    try:
        if predictor.model is None:
            raise HTTPException(status_code=503, detail="Prediction model not loaded")
        
        # Convert sensor data to DataFrame
        sensor_df = pd.DataFrame([request.sensor_data])
        
        # Make prediction
        result = predictor.predict(sensor_df)
        result['equipment_id'] = request.equipment_id
        
        logger.info(f"Prediction made for equipment {request.equipment_id}: {result['failure_prob']:.2%}")
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health/{equipment_id}", response_model=HealthCheckResponse)
async def get_equipment_health(equipment_id: str):
    """
    Get current equipment health status
    """
    try:
        # In production, this would query the database
        # For now, return mock data
        return HealthCheckResponse(
            equipment_id=equipment_id,
            status="operational",
            failure_probability=0.15,
            last_maintenance="2024-11-15T10:00:00",
            next_maintenance="2025-01-15T10:00:00",
            anomalies_detected=0
        )
    
    except Exception as e:
        logger.error(f"Error getting equipment health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/data/ingest")
async def ingest_sensor_data(request: DataIngestionRequest, background_tasks: BackgroundTasks):
    """
    Ingest new sensor data
    """
    try:
        logger.info(f"Ingesting {len(request.readings)} readings for equipment {request.equipment_id}")
        
        # In production, this would:
        # 1. Validate data
        # 2. Store in database
        # 3. Trigger real-time predictions
        # 4. Check for anomalies
        
        # Add background task for processing
        background_tasks.add_task(process_sensor_data, request.equipment_id, request.readings)
        
        return {
            "status": "accepted",
            "equipment_id": request.equipment_id,
            "readings_received": len(request.readings),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Data ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/alerts", response_model=List[AlertResponse])
async def get_active_alerts(severity: Optional[str] = None, limit: int = 100):
    """
    Get active alerts
    """
    try:
        # In production, query database for active alerts
        # Mock response for now
        alerts = [
            AlertResponse(
                alert_id="ALT001",
                equipment_id="EQ001",
                severity="HIGH",
                message="High failure probability detected",
                timestamp=datetime.now().isoformat(),
                acknowledged=False
            )
        ]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity.upper()]
        
        return alerts[:limit]
    
    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/maintenance/schedule")
async def get_maintenance_schedule(days: int = 30):
    """
    Get optimized maintenance schedule
    """
    try:
        # In production, this would use optimization algorithms
        # Mock response for now
        return {
            "schedule": [
                {
                    "equipment_id": "EQ001",
                    "scheduled_date": "2024-12-15T09:00:00",
                    "priority": "high",
                    "estimated_duration": "4 hours"
                }
            ],
            "total_items": 1,
            "period_days": days
        }
    
    except Exception as e:
        logger.error(f"Error getting maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/anomaly/detect")
async def detect_anomaly(request: PredictionRequest):
    """
    Detect anomalies in sensor data
    """
    try:
        if anomaly_detector.model is None:
            raise HTTPException(status_code=503, detail="Anomaly detection model not loaded")
        
        # Convert sensor data to DataFrame
        sensor_df = pd.DataFrame([request.sensor_data])
        
        # Detect anomalies
        result = anomaly_detector.detect(sensor_df)
        result['equipment_id'] = request.equipment_id
        result['timestamp'] = datetime.now().isoformat()
        
        return result
    
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_sensor_data(equipment_id: str, readings: List[SensorReading]):
    """
    Background task to process sensor data
    """
    logger.info(f"Processing sensor data for {equipment_id}")
    # Implement data processing logic here
    pass


if __name__ == "__main__":
    import uvicorn
    
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8000)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=config.get('api.reload', False)
    )
