"""
Model Training Module for Predictive Maintenance
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model training class for failure prediction
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize trainer
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = {}
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'failure',
        test_size: float = 0.2,
        balance_data: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test data
            balance_data: Whether to balance classes using SMOTE
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Balance classes if requested
        if balance_data:
            logger.info("Balancing classes using SMOTE")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
        logger.info(f"Training data shape: {X_train_scaled.shape}")
        logger.info(f"Test data shape: {X_test_scaled.shape}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameter_tuning: bool = False,
        cv_folds: int = 5
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Initialize model
        if self.model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'xgboost':
            base_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            self.training_history['best_params'] = grid_search.best_params_
        else:
            self.model = base_model
            self.model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
        logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.training_history['cv_scores'] = cv_scores.tolist()
        self.training_history['training_date'] = datetime.now().isoformat()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        results = {
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }
        
        self.training_history['evaluation'] = results
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            return importance_df.head(top_n)
        else:
            logger.warning("Model does not support feature importance")
            return None
    
    def save_model(self, model_path: str):
        """
        Save trained model
        
        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metadata': {
                'model_type': self.model_type,
                'training_history': self.training_history
            }
        }
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train predictive maintenance model')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--model-type', default='xgboost', choices=['random_forest', 'xgboost', 'gradient_boosting'])
    parser.add_argument('--output', default='models/trained/failure_prediction.pkl', help='Output model path')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--target', default='failure', help='Target column name')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Initialize trainer
    trainer = ModelTrainer(model_type=args.model_type)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(data, target_column=args.target)
    
    # Train model
    trainer.train(X_train, y_train, hyperparameter_tuning=args.tune)
    
    # Evaluate
    results = trainer.evaluate(X_test, y_test)
    
    # Feature importance
    feature_importance = trainer.get_feature_importance()
    if feature_importance is not None:
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    # Save model
    trainer.save_model(args.output)
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
