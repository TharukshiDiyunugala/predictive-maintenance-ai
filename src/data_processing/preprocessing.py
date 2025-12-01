"""
Data Preprocessing Module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for sensor data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.interpolation_method = self.config.get('interpolation_method', 'linear')
        self.max_missing_ratio = self.config.get('max_missing_ratio', 0.1)
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean sensor data
        
        Args:
            data: Raw sensor data
        
        Returns:
            Cleaned data
        """
        logger.info(f"Cleaning data with {len(data)} rows")
        
        # Make a copy
        df = data.copy()
        
        # Check missing values
        df = self._handle_missing_values(df)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        logger.info(f"Cleaned data has {len(df)} rows")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Check missing ratio
        missing_ratio = df.isnull().sum() / len(df)
        
        # Drop columns with too many missing values
        cols_to_drop = missing_ratio[missing_ratio > self.max_missing_ratio].index
        if len(cols_to_drop) > 0:
            logger.warning(f"Dropping columns with >10% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
        
        # Interpolate remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(
            method=self.interpolation_method,
            limit_direction='both'
        )
        
        # Fill any remaining NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        
        if before > after:
            logger.info(f"Removed {before - after} duplicate rows")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from numeric columns"""
        if not self.config.get('remove_outliers', True):
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if self.outlier_method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower) | (df[col] > upper)
                df = df[~outliers]
        
        elif self.outlier_method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < self.outlier_threshold]
        
        return df
    
    def normalize_timestamps(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        freq: str = '1min'
    ) -> pd.DataFrame:
        """
        Normalize timestamps to regular intervals
        
        Args:
            data: Input data
            timestamp_col: Name of timestamp column
            freq: Resampling frequency
        
        Returns:
            Data with normalized timestamps
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Set as index
        df = df.set_index(timestamp_col)
        
        # Resample to regular intervals
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_resampled = df[numeric_cols].resample(freq).mean()
        
        # Interpolate missing values
        df_resampled = df_resampled.interpolate(method='time')
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality
        
        Args:
            data: Data to validate
        
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_ratio': (data.isnull().sum() / len(data)).to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.astype(str).to_dict()
        }
        
        # Check for constant columns
        constant_cols = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        quality_report['constant_columns'] = constant_cols
        
        # Basic statistics
        quality_report['numeric_summary'] = data.describe().to_dict()
        
        return quality_report


class SensorDataLoader:
    """
    Load sensor data from various sources
    """
    
    @staticmethod
    def load_from_csv(
        file_path: str,
        timestamp_col: str = 'timestamp',
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            timestamp_col: Name of timestamp column
            parse_dates: Whether to parse dates
        
        Returns:
            DataFrame with sensor data
        """
        logger.info(f"Loading data from {file_path}")
        
        parse_dates_list = [timestamp_col] if parse_dates else False
        
        df = pd.read_csv(file_path, parse_dates=parse_dates_list)
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    @staticmethod
    def load_from_database(
        query: str,
        connection_string: str
    ) -> pd.DataFrame:
        """
        Load data from database
        
        Args:
            query: SQL query
            connection_string: Database connection string
        
        Returns:
            DataFrame with sensor data
        """
        from sqlalchemy import create_engine
        
        logger.info("Loading data from database")
        engine = create_engine(connection_string)
        
        df = pd.read_sql(query, engine)
        
        logger.info(f"Loaded {len(df)} rows from database")
        
        return df
    
    @staticmethod
    def save_to_csv(data: pd.DataFrame, file_path: str):
        """
        Save data to CSV file
        
        Args:
            data: DataFrame to save
            file_path: Output file path
        """
        data.to_csv(file_path, index=False)
        logger.info(f"Saved {len(data)} rows to {file_path}")
