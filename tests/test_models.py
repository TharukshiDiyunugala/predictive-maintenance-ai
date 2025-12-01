"""
Unit Tests for Predictor Module
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import MaintenancePredictor


class TestMaintenancePredictor(unittest.TestCase):
    """Test cases for MaintenancePredictor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = MaintenancePredictor()
        
        # Create sample data
        self.sample_data = pd.DataFrame([{
            'temperature': 75.0,
            'vibration': 5.0,
            'pressure': 98.0,
            'current': 10.5,
            'voltage': 235.0,
            'rpm': 2950,
            'power': 2467.5,
            'efficiency': 99.5
        }])
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        self.assertIsNone(self.predictor.model)
        self.assertIsNone(self.predictor.feature_names)
    
    def test_estimate_time_to_failure(self):
        """Test time to failure estimation"""
        # High probability
        days = self.predictor._estimate_time_to_failure(0.95)
        self.assertEqual(days, 1)
        
        # Medium probability
        days = self.predictor._estimate_time_to_failure(0.75)
        self.assertEqual(days, 7)
        
        # Low probability
        days = self.predictor._estimate_time_to_failure(0.3)
        self.assertIsNone(days)
    
    def test_get_severity_level(self):
        """Test severity level determination"""
        self.assertEqual(self.predictor._get_severity_level(0.95), "CRITICAL")
        self.assertEqual(self.predictor._get_severity_level(0.75), "HIGH")
        self.assertEqual(self.predictor._get_severity_level(0.55), "MEDIUM")
        self.assertEqual(self.predictor._get_severity_level(0.3), "LOW")
    
    def test_get_recommendations(self):
        """Test recommendations generation"""
        recs = self.predictor._get_recommendations(0.95, 1)
        self.assertIn("URGENT", recs[0])
        
        recs = self.predictor._get_recommendations(0.3, None)
        self.assertIn("No immediate action", recs[0])


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        from data_processing import DataPreprocessor
        self.preprocessor = DataPreprocessor()
        
        # Create sample data with issues
        self.sample_data = pd.DataFrame({
            'temperature': [70, 75, np.nan, 80, 85],
            'vibration': [3, 3.5, 4, 4.5, 5],
            'pressure': [100, 100, 100, 100, 100]  # Constant column
        })
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        clean_data = self.preprocessor._handle_missing_values(self.sample_data)
        self.assertFalse(clean_data.isnull().any().any())
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        data_with_dupes = pd.concat([self.sample_data, self.sample_data.iloc[[0]]])
        clean_data = self.preprocessor._remove_duplicates(data_with_dupes)
        self.assertEqual(len(clean_data), len(self.sample_data))


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer"""
    
    def setUp(self):
        """Set up test fixtures"""
        from features import FeatureEngineer
        self.engineer = FeatureEngineer()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'temperature': np.random.randn(100) + 70,
            'vibration': np.random.randn(100) + 4,
            'pressure': np.random.randn(100) + 100,
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H')
        })
    
    def test_create_features(self):
        """Test feature creation"""
        sensor_cols = ['temperature', 'vibration', 'pressure']
        features = self.engineer.create_features(self.sample_data, sensor_cols)
        
        # Should have more features than original
        self.assertGreater(len(features.columns), len(sensor_cols))
    
    def test_time_features(self):
        """Test time-based features"""
        features = self.engineer._add_time_features(self.sample_data)
        self.assertIn('hour', features.columns)
        self.assertIn('day_of_week', features.columns)


if __name__ == '__main__':
    unittest.main()
