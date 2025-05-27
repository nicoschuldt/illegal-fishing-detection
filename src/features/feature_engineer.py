"""
Feature Engineering Module

Converts AIS trip data into the exact feature set required by the trained model.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from models import AISData

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Converts AIS data into features for the fishing detection model."""
    
    def __init__(self):
        """Initialize the feature engineer with the required feature set."""
        # These are the exact 15 features selected by the final model
        self.required_features = [
            'speed_min', 'course_changes_per_hour', 'speed_q25', 'speed_consistency',
            'speed_std', 'course_diff_std', 'area_covered', 'distance_from_port_q25',
            'distance_from_shore_q25', 'distance_from_shore_min', 'lon_range',
            'distance_from_port_min', 'course_diff_max', 'course_diff_mean', 'speed_mean'
        ]
    
    def engineer_features(self, ais_data: AISData) -> Optional[pd.DataFrame]:
        """
        Engineer features for a single AIS dataset.
        
        Parameters:
        -----------
        ais_data : AISData
            Preprocessed AIS data (could be full dataset or single trip)
            
        Returns:
        --------
        pd.DataFrame with one row per AIS point and engineered features
        """
        try:
            logger.info(f"Engineering features for {len(ais_data.points)} AIS points")
            
            # Convert to DataFrame for processing
            df = ais_data.to_dataframe()
            
            if len(df) < 3:
                logger.error("Insufficient data points for feature engineering")
                return None
            
            # Sort by time to ensure proper sequence
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate movement features
            df = self._calculate_movement_features(df)
            
            # Calculate rolling window features
            df = self._calculate_rolling_features(df)
            
            # Calculate spatial features  
            df = self._calculate_spatial_features(df)
            
            # Select only the required features
            feature_df = df[self.required_features].copy()
            
            # Handle any remaining missing values
            feature_df = self._handle_missing_values(feature_df)
            
            logger.info(f"Successfully engineered {len(feature_df.columns)} features")
            return feature_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            return None
    
    def _calculate_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate movement-based features."""
        # Calculate time differences
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600  # hours
        
        # Calculate course changes
        df['course_diff'] = df['course'].diff().abs()
        # Handle circular nature of course (e.g., 350° to 10° should be 20°, not 340°)
        df['course_diff'] = np.minimum(df['course_diff'], 360 - df['course_diff'])
        
        # Calculate coordinates differences for spatial features
        df['lat_diff'] = df['lat'].diff()
        df['lon_diff'] = df['lon'].diff()
        
        # Calculate point-to-point distance (simplified)
        df['point_distance'] = np.sqrt(df['lat_diff']**2 + df['lon_diff']**2)
        
        return df
    
    def _calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling window features for the entire dataset."""
        # For the pipeline, we treat the entire AIS dataset as one "trip"
        # and calculate trip-level statistics
        
        # Speed features
        df['speed_mean'] = df['speed'].mean()
        df['speed_min'] = df['speed'].min()
        df['speed_std'] = df['speed'].std()
        df['speed_q25'] = df['speed'].quantile(0.25)
        
        # Speed consistency (1 - coefficient of variation)
        cv = df['speed'].std() / df['speed'].mean() if df['speed'].mean() > 0 else 0
        df['speed_consistency'] = 1.0 - cv
        
        # Course change features
        df['course_diff_mean'] = df['course_diff'].mean()
        df['course_diff_max'] = df['course_diff'].max()
        df['course_diff_std'] = df['course_diff'].std()
        
        # Calculate course changes per hour
        total_time = df['time_diff'].sum()
        major_course_changes = (df['course_diff'] > 30).sum()
        df['course_changes_per_hour'] = major_course_changes / total_time if total_time > 0 else 0
        
        return df
    
    def _calculate_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate spatial context features."""
        # Distance features
        df['distance_from_shore_min'] = df['distance_from_shore'].min()
        df['distance_from_shore_q25'] = df['distance_from_shore'].quantile(0.25)
        
        df['distance_from_port_min'] = df['distance_from_port'].min()
        df['distance_from_port_q25'] = df['distance_from_port'].quantile(0.25)
        
        # Geographic range features
        df['lon_range'] = df['lon'].max() - df['lon'].min()
        lat_range = df['lat'].max() - df['lat'].min()
        
        # Area covered (rough approximation)
        df['area_covered'] = lat_range * df['lon_range']
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle any missing values in the feature set."""
        for col in df.columns:
            if df[col].isna().any():
                if col in ['speed_consistency', 'course_changes_per_hour']:
                    # For rates and ratios, use 0 as default
                    df[col] = df[col].fillna(0.0)
                else:
                    # For other features, use median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)
        
        return df
    
    def validate_features(self, features_df: pd.DataFrame) -> bool:
        """
        Validate that the engineered features are ready for the model.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Engineered features
            
        Returns:
        --------
        bool: True if features are valid
        """
        # Check that all required features are present
        missing_features = set(self.required_features) - set(features_df.columns)
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False
        
        # Check for any infinite or NaN values
        if features_df.isna().any().any():
            logger.error("Features contain NaN values")
            return False
        
        if np.isinf(features_df.values).any():
            logger.error("Features contain infinite values")
            return False
        
        # Check for reasonable value ranges
        if (features_df < 0).any().any():
            negative_features = features_df.columns[(features_df < 0).any()].tolist()
            logger.warning(f"Features with negative values: {negative_features}")
        
        return True


# Trip-level feature engineering for batch processing
class TripFeatureEngineer:
    """Engineers features at the trip level for batch processing."""
    
    def __init__(self):
        """Initialize the trip feature engineer."""
        self.feature_engineer = FeatureEngineer()
    
    def engineer_trip_features(self, trips: List[AISData]) -> pd.DataFrame:
        """
        Engineer features for multiple trips.
        
        Parameters:
        -----------
        trips : List[AISData]
            List of individual trip data
            
        Returns:
        --------
        pd.DataFrame: Features for each trip (one row per trip)
        """
        trip_features = []
        
        for i, trip in enumerate(trips):
            try:
                # Engineer features for this trip
                features = self.feature_engineer.engineer_features(trip)
                
                if features is None:
                    logger.warning(f"Failed to engineer features for trip {i}")
                    continue
                
                # For trip-level features, we take the mean of all points in the trip
                # (since the trip-level features are constant across all points in the trip)
                trip_feature_row = features.iloc[0].to_dict()
                
                # Add trip metadata
                trip_feature_row.update({
                    'trip_index': i,
                    'mmsi': trip.mmsi,
                    'start_time': trip.start_time,
                    'end_time': trip.end_time,
                    'duration_hours': (trip.end_time - trip.start_time).total_seconds() / 3600,
                    'point_count': len(trip.points)
                })
                
                trip_features.append(trip_feature_row)
                
            except Exception as e:
                logger.error(f"Failed to process trip {i}: {str(e)}")
                continue
        
        if not trip_features:
            logger.error("No trips could be processed")
            return pd.DataFrame()
        
        return pd.DataFrame(trip_features)


def engineer_ais_features(ais_data: AISData) -> Optional[pd.DataFrame]:
    """
    Convenience function to engineer features from AIS data.
    
    Parameters:
    -----------
    ais_data : AISData
        Preprocessed AIS data
        
    Returns:
    --------
    pd.DataFrame with engineered features or None if failed
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(ais_data)