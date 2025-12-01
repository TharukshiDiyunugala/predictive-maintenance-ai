"""
Remaining Useful Life (RUL) Prediction using LSTM
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class RULPredictor:
    """
    Remaining Useful Life prediction using LSTM
    """
    
    def __init__(self, sequence_length: int = 50):
        """
        Initialize RUL predictor
        
        Args:
            sequence_length: Length of input sequences
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = None
    
    def build_model(self, n_features: int):
        """
        Build LSTM model
        
        Args:
            n_features: Number of input features
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # RUL prediction
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info(f"LSTM model built with {n_features} features")
        return model
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_column: str = 'rul'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: Input DataFrame
            target_column: Name of target column (RUL)
        
        Returns:
            X, y arrays
        """
        self.feature_names = [col for col in data.columns if col != target_column]
        
        # Scale features
        features = data[self.feature_names].values
        target = data[target_column].values
        
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(features_scaled[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            n_features = X_train.shape[2]
            self.build_model(n_features)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM model training completed")
        return history
    
    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict remaining useful life
        
        Args:
            sequence: Input sequence
        
        Returns:
            Predicted RUL in days
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Scale input
        sequence_scaled = self.scaler.transform(sequence)
        
        # Reshape for prediction
        sequence_reshaped = sequence_scaled.reshape(1, self.sequence_length, -1)
        
        # Predict
        rul = self.model.predict(sequence_reshaped, verbose=0)[0][0]
        
        return max(0, float(rul))  # RUL cannot be negative
    
    def save_model(self, path: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        
        # Save scaler separately
        import joblib
        scaler_path = path.replace('.h5', '_scaler.pkl')
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }, scaler_path)
        
        logger.info(f"RUL model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        self.model = keras.models.load_model(path)
        
        # Load scaler
        import joblib
        scaler_path = path.replace('.h5', '_scaler.pkl')
        scaler_data = joblib.load(scaler_path)
        self.scaler = scaler_data['scaler']
        self.feature_names = scaler_data['feature_names']
        self.sequence_length = scaler_data['sequence_length']
        
        logger.info(f"RUL model loaded from {path}")
