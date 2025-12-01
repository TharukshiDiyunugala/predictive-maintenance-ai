"""
Feature Engineering Module
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from scipy import stats, signal
from scipy.fft import fft
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for sensor data
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.time_windows = self.config.get('time_windows', [5, 15, 30, 60])
    
    def create_features(self, data: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """
        Create features from raw sensor data
        
        Args:
            data: Input data with sensor readings
            sensor_cols: List of sensor column names
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating features...")
        
        df = data.copy()
        
        # Statistical features
        df = self._add_statistical_features(df, sensor_cols)
        
        # Time-based features
        if 'timestamp' in df.columns:
            df = self._add_time_features(df)
        
        # Rolling window features
        df = self._add_rolling_features(df, sensor_cols)
        
        # Rate of change features
        df = self._add_rate_of_change_features(df, sensor_cols)
        
        # Interaction features
        df = self._add_interaction_features(df, sensor_cols)
        
        logger.info(f"Created {len(df.columns)} total features")
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add statistical features"""
        for col in cols:
            # Basic statistics
            df[f'{col}_mean'] = df[col].rolling(window=10, min_periods=1).mean()
            df[f'{col}_std'] = df[col].rolling(window=10, min_periods=1).std()
            df[f'{col}_min'] = df[col].rolling(window=10, min_periods=1).min()
            df[f'{col}_max'] = df[col].rolling(window=10, min_periods=1).max()
            df[f'{col}_median'] = df[col].rolling(window=10, min_periods=1).median()
            
            # Percentiles
            df[f'{col}_q25'] = df[col].rolling(window=10, min_periods=1).quantile(0.25)
            df[f'{col}_q75'] = df[col].rolling(window=10, min_periods=1).quantile(0.75)
            
            # Range
            df[f'{col}_range'] = df[f'{col}_max'] - df[f'{col}_min']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' not in df.columns:
            return df
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['quarter'] = pd.to_datetime(df['timestamp']).dt.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add rolling window features"""
        for col in cols:
            for window in self.time_windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling max
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).max()
                
                # Rolling min
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).min()
        
        return df
    
    def _add_rate_of_change_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add rate of change features"""
        for col in cols:
            # First derivative (rate of change)
            df[f'{col}_rate_of_change'] = df[col].diff()
            
            # Second derivative (acceleration)
            df[f'{col}_acceleration'] = df[f'{col}_rate_of_change'].diff()
            
            # Cumulative sum
            df[f'{col}_cumsum'] = df[col].cumsum()
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add interaction features between sensors"""
        # Only create interactions for first few sensors to avoid explosion
        cols_subset = cols[:min(5, len(cols))]
        
        for i, col1 in enumerate(cols_subset):
            for col2 in cols_subset[i+1:]:
                # Product
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Ratio (avoid division by zero)
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)
        
        return df
    
    def add_fft_features(self, data: pd.DataFrame, cols: List[str], n_components: int = 5) -> pd.DataFrame:
        """
        Add FFT (Fourier Transform) features
        
        Args:
            data: Input data
            cols: Columns to transform
            n_components: Number of FFT components to keep
        
        Returns:
            DataFrame with FFT features
        """
        df = data.copy()
        
        for col in cols:
            # Apply FFT
            fft_values = fft(df[col].values)
            fft_magnitudes = np.abs(fft_values)
            
            # Keep top n components
            for i in range(n_components):
                df[f'{col}_fft_{i}'] = fft_magnitudes[i]
        
        return df
    
    def add_lag_features(
        self,
        data: pd.DataFrame,
        cols: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Add lag features
        
        Args:
            data: Input data
            cols: Columns to lag
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        df = data.copy()
        
        for col in cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def select_features(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        method: str = 'mutual_info',
        top_n: int = 50
    ) -> List[str]:
        """
        Select top features
        
        Args:
            data: Feature data
            target: Target variable
            method: Selection method
            top_n: Number of top features to select
        
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import mutual_info_classif, f_classif
        
        if method == 'mutual_info':
            scores = mutual_info_classif(data, target)
        elif method == 'f_classif':
            scores, _ = f_classif(data, target)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top features
        feature_scores = pd.DataFrame({
            'feature': data.columns,
            'score': scores
        }).sort_values('score', ascending=False)
        
        top_features = feature_scores.head(top_n)['feature'].tolist()
        
        logger.info(f"Selected top {len(top_features)} features")
        
        return top_features
