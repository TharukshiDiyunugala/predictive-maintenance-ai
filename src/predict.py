"""
Main Prediction Script
"""
import sys
from pathlib import Path
import argparse
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models import MaintenancePredictor
from data_processing import SensorDataLoader


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Predict equipment failure')
    parser.add_argument('--equipment-id', required=True, help='Equipment ID')
    parser.add_argument('--model', default='models/trained/failure_prediction.pkl', help='Model path')
    parser.add_argument('--data', help='Sensor data CSV file')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    predictor = MaintenancePredictor()
    predictor.load_model(args.model)
    
    # Load or create sample data
    if args.data:
        print(f"Loading data from {args.data}")
        loader = SensorDataLoader()
        data = loader.load_from_csv(args.data)
        
        # Filter for specific equipment if column exists
        if 'equipment_id' in data.columns:
            data = data[data['equipment_id'] == args.equipment_id]
        
        # Use latest reading
        sensor_data = data.tail(1)
    else:
        # Use sample data
        print("Using sample sensor data")
        sensor_data = pd.DataFrame([{
            'temperature': 75.0,
            'vibration': 5.0,
            'pressure': 98.0,
            'current': 10.5,
            'voltage': 235.0,
            'rpm': 2950,
            'power': 2467.5,
            'efficiency': 99.5
        }])
    
    # Make prediction
    print(f"\nPredicting failure for equipment: {args.equipment_id}")
    result = predictor.predict(sensor_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Equipment ID: {args.equipment_id}")
    print(f"Failure Probability: {result['failure_prob']:.2%}")
    print(f"Failure Predicted: {'YES' if result['failure_predicted'] else 'NO'}")
    print(f"Severity Level: {result['severity']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Days to Failure: {result['days_to_failure'] or 'N/A'}")
    print(f"\nRecommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec}")
    print("=" * 60)


if __name__ == '__main__':
    main()
