"""
Illegal Fishing Detection Pipeline

Main orchestration module that coordinates all components for detecting
illegal fishing activity from vessel AIS data.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Import our modules
from models import (
    AISData, FishingPredictions, AnalysisResult, FishingSummary,
    create_error_result, validate_mmsi, validate_time_window
)
from src.features.preprocessor import AISPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.predictor import FishingPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IllegalFishingDetector:
    """
    Main pipeline class for detecting illegal fishing activity.
    
    This class coordinates the ML components of the pipeline:
    1. AIS data preprocessing
    2. Feature engineering
    3. Fishing activity prediction
    
    The AIS data fetching and MPA analysis are handled by partner components.
    """
    
    def __init__(self, model_path: str = "models/fishing_classifier.joblib"):
        """
        Initialize the illegal fishing detector.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained ML model
        """
        try:
            self.preprocessor = AISPreprocessor()
            self.feature_engineer = FeatureEngineer()
            self.predictor = FishingPredictor(model_path)
            
            logger.info("Illegal Fishing Detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize detector: {str(e)}")
            raise
    
    def analyze_vessel_data(self, ais_dataframe: pd.DataFrame, mmsi: str) -> AnalysisResult:
        """
        Analyze vessel AIS data for fishing activity.
        
        This is the main ML pipeline that processes raw AIS data and returns
        fishing predictions. The AIS data should come from the partner's
        AIS fetcher component.
        
        Parameters:
        -----------
        ais_dataframe : pd.DataFrame
            Raw AIS data from the AIS fetcher component
        mmsi : str
            Vessel identifier
            
        Returns:
        --------
        AnalysisResult with fishing predictions (for MPA analyzer component)
        """
        try:
            logger.info(f"Starting ML analysis for vessel {mmsi}")
            
            # Validate inputs
            if not validate_mmsi(mmsi):
                return create_error_result(mmsi, "Invalid MMSI format")
            
            if ais_dataframe.empty:
                return create_error_result(mmsi, "Empty AIS data")
            
            # Step 1: Preprocess AIS data
            logger.info("Preprocessing AIS data...")
            ais_data = self.preprocessor.preprocess(ais_dataframe)
            
            if ais_data is None:
                return create_error_result(mmsi, "AIS data preprocessing failed")
            
            # Validate preprocessed data
            if not self.preprocessor.validate_ais_data(ais_data):
                return create_error_result(mmsi, "AIS data validation failed")
            
            logger.info(f"Preprocessed {len(ais_data.points)} AIS points")
            
            # Step 2: Engineer features
            logger.info("Engineering features...")
            features = self.feature_engineer.engineer_features(ais_data)
            
            if features is None:
                return create_error_result(mmsi, "Feature engineering failed")
            
            # Validate features
            if not self.feature_engineer.validate_features(features):
                return create_error_result(mmsi, "Feature validation failed")
            
            logger.info(f"Engineered {len(features.columns)} features")
            
            # Step 3: Predict fishing activity
            logger.info("Predicting fishing activity...")
            
            # Extract timestamps and coordinates for proper prediction
            timestamps = pd.Series([p.timestamp for p in ais_data.points])
            coordinates = pd.DataFrame({
                'lat': [p.lat for p in ais_data.points],
                'lon': [p.lon for p in ais_data.points]
            })
            
            predictions = self.predictor.predict_single_trip(
                features, mmsi, timestamps, coordinates
            )
            
            if predictions is None:
                return create_error_result(mmsi, "Fishing prediction failed")
            
            logger.info(f"Predicted {predictions.fishing_points_count} fishing points out of {predictions.total_points_count}")
            
            # Step 4: Create analysis result
            result = self._create_analysis_result(ais_data, predictions)
            
            logger.info(f"ML analysis complete for vessel {mmsi}")
            return result
            
        except Exception as e:
            logger.error(f"ML pipeline error for vessel {mmsi}: {str(e)}")
            return create_error_result(mmsi, f"Pipeline error: {str(e)}")
    
    def _create_analysis_result(self, ais_data: AISData, predictions: FishingPredictions) -> AnalysisResult:
        """Create analysis result from AIS data and predictions."""
        
        # Calculate fishing summary
        duration_hours = (ais_data.end_time - ais_data.start_time).total_seconds() / 3600
        avg_fishing_prob = sum(p.fishing_probability for p in predictions.predictions) / len(predictions.predictions)
        
        fishing_summary = FishingSummary(
            total_ais_points=predictions.total_points_count,
            fishing_points=predictions.fishing_points_count,
            fishing_percentage=predictions.fishing_percentage,
            avg_fishing_probability=avg_fishing_prob,
            time_period_hours=duration_hours
        )
        
        # Create analysis result (without MPA analysis - that's handled by partner)
        result = AnalysisResult(
            mmsi=predictions.mmsi,
            analysis_time=datetime.utcnow(),
            success=True,
            time_period_start=ais_data.start_time,
            time_period_end=ais_data.end_time,
            fishing_summary=fishing_summary,
            mpa_analysis=None,  # Will be filled by MPA analyzer component
            violation_detected=False,  # Will be determined by MPA analyzer
            risk_score=avg_fishing_prob,  # Base risk score from fishing probability
            risk_level=self._calculate_risk_level(avg_fishing_prob)
        )
        
        return result
    
    def _calculate_risk_level(self, avg_fishing_prob: float) -> str:
        """Calculate risk level from average fishing probability."""
        if avg_fishing_prob >= 0.8:
            return "HIGH"
        elif avg_fishing_prob >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.predictor.get_model_info()


# Interface functions for partner components
def process_ais_data_for_pipeline(ais_dataframe: pd.DataFrame, mmsi: str) -> Optional[FishingPredictions]:
    """
    Interface function for partner components to get fishing predictions.
    
    This function handles the ML pipeline and returns predictions that can
    be used by the MPA analyzer component.
    
    Parameters:
    -----------
    ais_dataframe : pd.DataFrame
        Raw AIS data (from AIS fetcher component)
    mmsi : str
        Vessel identifier
        
    Returns:
    --------
    FishingPredictions or None if processing fails
    """
    try:
        detector = IllegalFishingDetector()
        result = detector.analyze_vessel_data(ais_dataframe, mmsi)
        
        if not result.success:
            logger.error(f"Processing failed for {mmsi}: {result.error_message}")
            return None
        
       
        logger.info(f"Successfully processed {mmsi}")
        return result
        
    except Exception as e:
        logger.error(f"Interface function failed: {str(e)}")
        return None


def test_ml_pipeline():
    """Test the ML pipeline with sample data."""
    
    sample_data = {
        'mmsi': ['123456789'] * 20,
        'timestamp': [datetime.utcnow() - timedelta(hours=12) + timedelta(minutes=i*30) for i in range(20)],
        'lat': [14.8 + i*0.01 for i in range(20)],
        'lon': [-26.8 + i*0.01 for i in range(20)],
        'speed': [3.5 + (i % 3) for i in range(20)],
        'course': [230 + (i * 10) % 360 for i in range(20)],
        'distance_from_shore': [50000 + i*1000 for i in range(20)],
        'distance_from_port': [100000 + i*1000 for i in range(20)]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Test the pipeline
    detector = IllegalFishingDetector()
    result = detector.analyze_vessel_data(sample_df, "123456789")
    
    print("ML Pipeline Test Results:")
    print(f"Success: {result.success}")
    if result.success:
        print(f"Fishing points: {result.fishing_summary.fishing_points}")
        print(f"Total points: {result.fishing_summary.total_ais_points}")
        print(f"Fishing percentage: {result.fishing_summary.fishing_percentage:.1f}%")
        print(f"Risk level: {result.risk_level}")
    else:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    test_ml_pipeline()