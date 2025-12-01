"""
Main Predictor Module for Equipment Failure Prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MaintenancePredictor:
    """
    Main predictor class for equipment failure prediction
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model file
        """
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.model_metadata = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model from file
        
        Args:
            model_path: Path to model file
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', [])
            self.scaler = model_data.get('scaler', None)
            self.model_metadata = model_data.get('metadata', {})
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_model(self, model_path: str):
        """
        Save trained model to file
        
        Args:
            model_path: Path to save model
        """
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'metadata': self.model_metadata
        }
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def predict(self, sensor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict equipment failure probability
        
        Args:
            sensor_data: DataFrame with sensor readings
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare features
        features = self._prepare_features(sensor_data)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        failure_prob = self.model.predict_proba(features_scaled)[:, 1]
        failure_prediction = self.model.predict(features_scaled)
        
        # Calculate estimated time to failure
        days_to_failure = self._estimate_time_to_failure(failure_prob[0])
        
        # Determine severity level
        severity = self._get_severity_level(failure_prob[0])
        
        return {
            'failure_prob': float(failure_prob[0]),
            'failure_predicted': bool(failure_prediction[0]),
            'days_to_failure': days_to_failure,
            'severity': severity,
            'confidence': float(max(failure_prob[0], 1 - failure_prob[0])),
            'timestamp': datetime.now().isoformat(),
            'recommendations': self._get_recommendations(failure_prob[0], days_to_failure)
        }
    
    def _prepare_features(self, sensor_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from sensor data
        
        Args:
            sensor_data: Raw sensor data
        
        Returns:
            Feature array
        """
        if self.feature_names:
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(sensor_data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            return sensor_data[self.feature_names].values
        else:
            return sensor_data.values
    
    def _estimate_time_to_failure(self, failure_prob: float) -> Optional[int]:
        """
        Estimate days until failure based on probability
        
        Args:
            failure_prob: Failure probability
        
        Returns:
            Estimated days to failure
        """
        if failure_prob < 0.5:
            return None
        
        # Simple estimation: higher probability = sooner failure
        # In production, use more sophisticated RUL models
        if failure_prob >= 0.9:
            return 1
        elif failure_prob >= 0.8:
            return 3
        elif failure_prob >= 0.7:
            return 7
        else:
            return 14
    
    def _get_severity_level(self, failure_prob: float) -> str:
        """
        Determine severity level
        
        Args:
            failure_prob: Failure probability
        
        Returns:
            Severity level string
        """
        if failure_prob >= 0.9:
            return "CRITICAL"
        elif failure_prob >= 0.7:
            return "HIGH"
        elif failure_prob >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommendations(self, failure_prob: float, days_to_failure: Optional[int]) -> list:
        """
        Generate maintenance recommendations
        
        Args:
            failure_prob: Failure probability
            days_to_failure: Estimated days to failure
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if failure_prob >= 0.9:
            recommendations.append("URGENT: Schedule immediate maintenance")
            recommendations.append("Reduce equipment load until maintenance completed")
            recommendations.append("Prepare replacement parts")
        elif failure_prob >= 0.7:
            recommendations.append("Schedule maintenance within next 7 days")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Verify spare parts availability")
        elif failure_prob >= 0.5:
            recommendations.append("Plan maintenance within next 2 weeks")
            recommendations.append("Continue normal monitoring")
        else:
            recommendations.append("No immediate action required")
            recommendations.append("Continue regular maintenance schedule")
        
        return recommendations
    
    def batch_predict(self, equipment_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Predict for multiple equipment items
        
        Args:
            equipment_data: Dictionary mapping equipment_id to sensor data
        
        Returns:
            Dictionary mapping equipment_id to predictions
        """
        results = {}
        for equipment_id, sensor_data in equipment_data.items():
            try:
                results[equipment_id] = self.predict(sensor_data)
            except Exception as e:
                logger.error(f"Error predicting for {equipment_id}: {str(e)}")
                results[equipment_id] = {'error': str(e)}
        
        return results
