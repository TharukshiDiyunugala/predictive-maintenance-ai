"""
Example: Training and Using the Predictor
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import MaintenancePredictor
from models.train import ModelTrainer
from data_processing import SensorDataLoader, DataPreprocessor
from features import FeatureEngineer
import pandas as pd


def main():
    """Example workflow"""
    
    print("=" * 60)
    print("Predictive Maintenance - Training Example")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading sample data...")
    loader = SensorDataLoader()
    data = loader.load_from_csv('data/sample_sensor_data.csv')
    print(f"   Loaded {len(data)} samples")
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    # Don't remove outliers - they indicate failures!
    preprocessor = DataPreprocessor(config={'remove_outliers': False})
    clean_data = preprocessor.clean_data(data)
    print(f"   Clean data: {len(clean_data)} samples")
    
    # 3. Feature engineering
    print("\n3. Engineering features...")
    engineer = FeatureEngineer()
    sensor_cols = ['temperature', 'vibration', 'pressure', 'current', 'voltage', 'rpm']
    
    # Add timestamp column if not present
    if 'timestamp' not in clean_data.columns:
        clean_data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(clean_data), freq='1H')
    
    feature_data = engineer.create_features(clean_data, sensor_cols)
    
    # Drop non-numeric columns (like equipment_id, timestamp) BEFORE dropping NaNs
    numeric_cols = feature_data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    feature_data = feature_data[numeric_cols]
    
    # Check class distribution before dropping NaNs
    print(f"   Before dropna - Failure rate: {feature_data['failure'].mean():.2%}")
    
    # Drop rows with NaN (from rolling windows)
    feature_data = feature_data.dropna()
    
    # Check class distribution after dropping NaNs
    print(f"   After dropna - Failure rate: {feature_data['failure'].mean():.2%}")
    print(f"   Samples remaining: {len(feature_data)}")
    print(f"   Created {len(feature_data.columns)} features")
    
    # 4. Train model
    print("\n4. Training model...")
    trainer = ModelTrainer(model_type='xgboost')
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        feature_data,
        target_column='failure'
    )
    
    # Train
    trainer.train(X_train, y_train, hyperparameter_tuning=False)
    
    # Evaluate
    print("\n5. Evaluating model...")
    results = trainer.evaluate(X_test, y_test)
    
    print(f"\n   Model Performance:")
    print(f"   - Accuracy: {results['accuracy']:.4f}")
    print(f"   - Precision: {results['precision']:.4f}")
    print(f"   - Recall: {results['recall']:.4f}")
    print(f"   - F1 Score: {results['f1_score']:.4f}")
    print(f"   - ROC-AUC: {results['roc_auc']:.4f}")
    
    # Feature importance
    print("\n6. Top 10 Most Important Features:")
    importance = trainer.get_feature_importance(top_n=10)
    print(importance.to_string(index=False))
    
    # 7. Save model
    print("\n7. Saving model...")
    import os
    os.makedirs('models/trained', exist_ok=True)
    trainer.save_model('models/trained/failure_prediction.pkl')
    print("   Model saved to models/trained/failure_prediction.pkl")
    
    # 8. Test prediction
    print("\n8. Testing prediction...")
    predictor = MaintenancePredictor()
    predictor.load_model('models/trained/failure_prediction.pkl')
    
    # Use first row of test data
    test_sample = pd.DataFrame([X_test[0]], columns=trainer.feature_names)
    prediction = predictor.predict(test_sample)
    
    print(f"\n   Sample Prediction:")
    print(f"   - Failure Probability: {prediction['failure_prob']:.2%}")
    print(f"   - Severity: {prediction['severity']}")
    print(f"   - Days to Failure: {prediction['days_to_failure']}")
    print(f"   - Recommendations: {', '.join(prediction['recommendations'])}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
