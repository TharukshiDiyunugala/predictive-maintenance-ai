"""
Anomaly Detection Module
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
import joblib
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detection for sensor data
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.threshold = None
    
    def fit(self, data: pd.DataFrame):
        """
        Train anomaly detection model
        
        Args:
            data: Training data
        """
        self.feature_names = list(data.columns)
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        # Fit model
        self.model.fit(data_scaled)
        
        # Calculate anomaly scores for threshold
        scores = self.model.score_samples(data_scaled)
        self.threshold = np.percentile(scores, 5)
        
        logger.info(f"Anomaly detector trained on {len(data)} samples")
    
    def detect(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies in data
        
        Args:
            data: Sensor data to check
        
        Returns:
            Dictionary with anomaly detection results
        """
        if self.feature_names:
            data = data[self.feature_names]
        
        # Scale data
        data_scaled = self.scaler.transform(data)
        
        # Predict
        predictions = self.model.predict(data_scaled)
        anomaly_scores = self.model.score_samples(data_scaled)
        
        # -1 for anomalies, 1 for normal
        is_anomaly = predictions == -1
        
        return {
            'is_anomaly': bool(is_anomaly[0]) if len(is_anomaly) == 1 else is_anomaly.tolist(),
            'anomaly_score': float(anomaly_scores[0]) if len(anomaly_scores) == 1 else anomaly_scores.tolist(),
            'severity': self._get_anomaly_severity(anomaly_scores[0]) if len(anomaly_scores) == 1 else None,
            'num_anomalies': int(np.sum(is_anomaly)),
            'total_samples': len(data)
        }
    
    def _get_anomaly_severity(self, score: float) -> str:
        """Get anomaly severity level"""
        if score < self.threshold * 1.5:
            return "CRITICAL"
        elif score < self.threshold * 1.2:
            return "HIGH"
        elif score < self.threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def save_model(self, path: str):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.threshold
        }
        joblib.dump(model_data, path)
        logger.info(f"Anomaly detector saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data['threshold']
        logger.info(f"Anomaly detector loaded from {path}")
