"""
Monitoring and Logging Module
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI
from fastapi.responses import Response


# Prometheus metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['equipment_id', 'result']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['equipment_id']
)

failure_probability = Gauge(
    'equipment_failure_probability',
    'Current failure probability for equipment',
    ['equipment_id']
)

anomaly_counter = Counter(
    'anomalies_detected_total',
    'Total number of anomalies detected',
    ['equipment_id', 'severity']
)

api_requests = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


def setup_logging(
    log_file: str = 'logs/system.log',
    log_level: str = 'INFO',
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        max_bytes: Maximum log file size
        backup_count: Number of backup files
    
    Returns:
        Configured logger
    """
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('predictive_maintenance')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(JSONFormatter())
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


class MetricsCollector:
    """
    Collect and manage application metrics
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = {
            'predictions': 0,
            'anomalies': 0,
            'api_calls': 0,
            'errors': 0
        }
    
    def record_prediction(self, equipment_id: str, result: Dict[str, Any], latency: float):
        """
        Record prediction metrics
        
        Args:
            equipment_id: Equipment identifier
            result: Prediction result
            latency: Prediction latency in seconds
        """
        prediction_counter.labels(
            equipment_id=equipment_id,
            result='failure' if result.get('failure_predicted') else 'normal'
        ).inc()
        
        prediction_latency.labels(equipment_id=equipment_id).observe(latency)
        
        failure_probability.labels(equipment_id=equipment_id).set(
            result.get('failure_prob', 0)
        )
        
        self.metrics['predictions'] += 1
    
    def record_anomaly(self, equipment_id: str, severity: str):
        """
        Record anomaly detection
        
        Args:
            equipment_id: Equipment identifier
            severity: Anomaly severity
        """
        anomaly_counter.labels(
            equipment_id=equipment_id,
            severity=severity
        ).inc()
        
        self.metrics['anomalies'] += 1
    
    def record_api_call(self, endpoint: str, method: str, status: int):
        """
        Record API call
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status: HTTP status code
        """
        api_requests.labels(
            endpoint=endpoint,
            method=method,
            status=str(status)
        ).inc()
        
        self.metrics['api_calls'] += 1
        
        if status >= 400:
            self.metrics['errors'] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary
        
        Returns:
            Dictionary with metrics
        """
        return self.metrics.copy()


def add_monitoring_endpoints(app: FastAPI):
    """
    Add monitoring endpoints to FastAPI app
    
    Args:
        app: FastAPI application
    """
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )
    
    @app.get("/api/v1/monitoring/stats")
    async def get_stats():
        """Get application statistics"""
        collector = MetricsCollector()
        return collector.get_metrics_summary()


# Global logger instance
logger = setup_logging()
