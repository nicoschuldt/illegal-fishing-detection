"""
AIS Data Preprocessing Module

Handles cleaning and validation of raw AIS data for the fishing detection pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import logging

from models import AISData, create_error_result

logger = logging.getLogger(__name__)

class AISPreprocessor:
    """Preprocesses raw AIS data for feature engineering."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.required_columns = [
            'mmsi', 'timestamp', 'lat', 'lon', 'speed', 
            'course', 'distance_from_shore', 'distance_from_port'
        ]
    
    def preprocess(self, df: pd.DataFrame) -> Optional[AISData]:
        """
        Preprocess raw AIS DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw AIS data from API or file
            
        Returns:
        --------
        AISData or None if preprocessing fails
        """
        try:
            logger.info(f"Preprocessing AIS data with {len(df)} points")
            
            # Validate input
            if not self._validate_input(df):
                return None
            
            # Clean the data
            cleaned_df = self._clean_data(df.copy())
            
            if cleaned_df is None or len(cleaned_df) < 5:
                logger.error("Insufficient data after cleaning")
                return None
            
            # Convert to AISData model
            ais_data = AISData.from_dataframe(cleaned_df)
            
            logger.info(f"Successfully preprocessed {len(ais_data.points)} AIS points")
            return ais_data
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return None
    
    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame has required structure."""
        if df.empty:
            logger.error("Empty DataFrame provided")
            return False
        
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        return True
    
    def _clean_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean and validate AIS data."""
        initial_rows = len(df)
        
        # 1. Handle timestamps
        df = self._clean_timestamps(df)
        if df is None:
            return None
        
        # 2. Clean coordinates
        df = self._clean_coordinates(df)
        
        # 3. Clean speed and course
        df = self._clean_navigation_data(df)
        
        # 4. Handle missing distance data
        df = self._clean_distance_data(df)
        
        # 5. Sort by time
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        final_rows = len(df)
        logger.info(f"Cleaned data: {initial_rows} → {final_rows} points")
        
        return df if final_rows > 0 else None
    
    def _clean_timestamps(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean and convert timestamps."""
        try:
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                # Try unix timestamp first
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except (ValueError, OSError):
                    # Try direct conversion
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Remove rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Timestamp cleaning failed: {str(e)}")
            return None
    
    def _clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean latitude and longitude data."""
        # Remove rows with missing coordinates
        df = df.dropna(subset=['lat', 'lon'])
        
        valid_coords = (
            (df['lat'] >= -90) & (df['lat'] <= 90) &
            (df['lon'] >= -180) & (df['lon'] <= 180)
        )
        
        invalid_count = (~valid_coords).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} points with invalid coordinates")
            df = df[valid_coords]
        
        return df
    
    def _clean_navigation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean speed and course data."""
        # Clean speed: must be non-negative, reasonable upper bound
        df.loc[df['speed'] < 0, 'speed'] = np.nan
        df.loc[df['speed'] > 100, 'speed'] = np.nan  # 100 knots is very fast for fishing vessels
        
        # Clean course: must be 0-360 degrees
        df.loc[(df['course'] < 0) | (df['course'] > 360), 'course'] = np.nan
        
        # Fill missing navigation data with reasonable defaults
        if df['speed'].isna().any():
            # Fill with median speed (conservative approach)
            median_speed = df['speed'].median()
            df['speed'] = df['speed'].fillna(median_speed if not np.isnan(median_speed) else 5.0)
        
        if df['course'].isna().any():
            # For course, we can't easily interpolate, so use forward fill
            df['course'] = df['course'].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _clean_distance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean distance from shore and port data."""
        # Ensure distances are non-negative
        df.loc[df['distance_from_shore'] < 0, 'distance_from_shore'] = np.nan
        df.loc[df['distance_from_port'] < 0, 'distance_from_port'] = np.nan
        
        # Fill missing distances with median (reasonable for fishing vessels)
        for col in ['distance_from_shore', 'distance_from_port']:
            if df[col].isna().any():
                median_dist = df[col].median()
                if not np.isnan(median_dist):
                    df[col] = df[col].fillna(median_dist)
                else:
                    # If all values are missing, use reasonable defaults
                    default_shore = 50000  # 50km from shore
                    default_port = 100000  # 100km from port
                    df[col] = df[col].fillna(default_shore if 'shore' in col else default_port)
        
        return df
    
    def validate_ais_data(self, ais_data: AISData) -> bool:
        """
        Validate that AIS data meets minimum requirements for feature engineering.
        
        Parameters:
        -----------
        ais_data : AISData
            Preprocessed AIS data
            
        Returns:
        --------
        bool: True if data is valid for feature engineering
        """
        if len(ais_data.points) < 5:
            logger.error("Insufficient AIS points for feature engineering (minimum 5 required)")
            return False
        
        # Check time span (should be reasonable for fishing detection)
        time_span = (ais_data.end_time - ais_data.start_time).total_seconds() / 3600
        if time_span < 0.5:  # Less than 30 minutes
            logger.warning("Very short time span for AIS data")
        
        # Check for reasonable geographic spread
        lats = [p.lat for p in ais_data.points]
        lons = [p.lon for p in ais_data.points]
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        if lat_range < 0.001 and lon_range < 0.001:  # Less than ~100m movement
            logger.warning("Very small geographic range in AIS data")
        
        return True


def preprocess_ais_dataframe(df: pd.DataFrame) -> Optional[AISData]:
    """
    Convenience function to preprocess AIS DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw AIS data
        
    Returns:
    --------
    AISData or None if preprocessing fails
    """
    preprocessor = AISPreprocessor()
    return preprocessor.preprocess(df)