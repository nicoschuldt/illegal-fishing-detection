"""
Trip Segmentation Module

Segments AIS data into individual trips based on time gaps between points.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import logging

from models import AISData

logger = logging.getLogger(__name__)

class TripSegmenter:
    """Segments AIS data into individual vessel trips."""
    
    def __init__(self, max_gap_hours: float = 4.0, min_trip_points: int = 5):
        """
        Initialize the trip segmenter.
        
        Parameters:
        -----------
        max_gap_hours : float
            Maximum time gap (hours) between points to consider same trip
        min_trip_points : int
            Minimum points required for a valid trip
        """
        self.max_gap_hours = max_gap_hours
        self.min_trip_points = min_trip_points
    
    def segment_trips(self, ais_data: AISData) -> List[AISData]:
        """
        Segment AIS data into individual trips.
        
        Parameters:
        -----------
        ais_data : AISData
            Preprocessed AIS data for a single vessel
            
        Returns:
        --------
        List[AISData]: List of individual trip data
        """
        try:
            logger.info(f"Segmenting {len(ais_data.points)} AIS points into trips")
            
            # Convert to DataFrame for easier processing
            df = ais_data.to_dataframe()
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate time differences
            df['time_diff_hours'] = df['timestamp'].diff().dt.total_seconds() / 3600
            
            # Identify trip boundaries
            df['new_trip'] = (df['time_diff_hours'] > self.max_gap_hours) | (df['time_diff_hours'].isna())
            
            # Assign trip IDs
            df['trip_id'] = df['new_trip'].cumsum()
            
            # Create individual trips
            trips = []
            for trip_id in df['trip_id'].unique():
                trip_df = df[df['trip_id'] == trip_id].copy()
                
                # Skip trips that are too short
                if len(trip_df) < self.min_trip_points:
                    logger.debug(f"Skipping trip {trip_id} with only {len(trip_df)} points")
                    continue
                
                # Create AISData for this trip
                trip_data = AISData.from_dataframe(trip_df.drop(['time_diff_hours', 'new_trip', 'trip_id'], axis=1))
                trips.append(trip_data)
            
            logger.info(f"Segmented data into {len(trips)} valid trips")
            return trips
            
        except Exception as e:
            logger.error(f"Trip segmentation failed: {str(e)}")
            return []
    
    def get_trip_summary(self, trips: List[AISData]) -> pd.DataFrame:
        """
        Get summary statistics for segmented trips.
        
        Parameters:
        -----------
        trips : List[AISData]
            List of trip data
            
        Returns:
        --------
        pd.DataFrame: Summary statistics for each trip
        """
        summaries = []
        
        for i, trip in enumerate(trips):
            duration_hours = (trip.end_time - trip.start_time).total_seconds() / 3600
            
            # Calculate basic statistics
            speeds = [p.speed for p in trip.points]
            avg_speed = sum(speeds) / len(speeds)
            
            distances_shore = [p.distance_from_shore for p in trip.points]
            avg_distance_shore = sum(distances_shore) / len(distances_shore)
            
            summary = {
                'trip_index': i,
                'mmsi': trip.mmsi,
                'start_time': trip.start_time,
                'end_time': trip.end_time,
                'duration_hours': duration_hours,
                'point_count': len(trip.points),
                'avg_speed': avg_speed,
                'avg_distance_from_shore': avg_distance_shore
            }
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def segment_ais_data(ais_data: AISData, max_gap_hours: float = 4.0, min_trip_points: int = 5) -> List[AISData]:
    """
    Convenience function to segment AIS data into trips.
    
    Parameters:
    -----------
    ais_data : AISData
        AIS data to segment
    max_gap_hours : float
        Maximum time gap between points for same trip
    min_trip_points : int
        Minimum points required for valid trip
        
    Returns:
    --------
    List[AISData]: Individual trips
    """
    segmenter = TripSegmenter(max_gap_hours, min_trip_points)
    return segmenter.segment_trips(ais_data)