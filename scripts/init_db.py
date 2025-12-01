"""
Database Initialization Script
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from config import config

Base = declarative_base()


class Equipment(Base):
    """Equipment table"""
    __tablename__ = 'equipment'
    
    id = Column(Integer, primary_key=True)
    equipment_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(200))
    type = Column(String(100))
    location = Column(String(200))
    installation_date = Column(DateTime)
    last_maintenance = Column(DateTime)
    status = Column(String(50), default='operational')
    created_at = Column(DateTime, default=datetime.utcnow)


class SensorReading(Base):
    """Sensor readings table"""
    __tablename__ = 'sensor_readings'
    
    id = Column(Integer, primary_key=True)
    equipment_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    temperature = Column(Float)
    vibration = Column(Float)
    pressure = Column(Float)
    current = Column(Float)
    voltage = Column(Float)
    rpm = Column(Float)
    power = Column(Float)
    efficiency = Column(Float)


class Prediction(Base):
    """Predictions table"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    equipment_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    failure_prob = Column(Float)
    failure_predicted = Column(Boolean)
    severity = Column(String(20))
    days_to_failure = Column(Integer)
    confidence = Column(Float)
    recommendations = Column(Text)


class Alert(Base):
    """Alerts table"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(String(50), unique=True)
    equipment_id = Column(String(50), nullable=False)
    severity = Column(String(20))
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)


class MaintenanceLog(Base):
    """Maintenance log table"""
    __tablename__ = 'maintenance_log'
    
    id = Column(Integer, primary_key=True)
    equipment_id = Column(String(50), nullable=False)
    maintenance_type = Column(String(100))
    description = Column(Text)
    scheduled_date = Column(DateTime)
    completed_date = Column(DateTime)
    technician = Column(String(100))
    status = Column(String(50), default='scheduled')
    notes = Column(Text)


def init_database():
    """Initialize database tables"""
    try:
        # Get database URL from config
        db_url = config.get_database_url()
        print(f"Connecting to database...")
        
        # Create engine
        engine = create_engine(db_url)
        
        # Create all tables
        print("Creating tables...")
        Base.metadata.create_all(engine)
        
        print("Database initialized successfully!")
        print("\nCreated tables:")
        print("  - equipment")
        print("  - sensor_readings")
        print("  - predictions")
        print("  - alerts")
        print("  - maintenance_log")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)


if __name__ == '__main__':
    init_database()
