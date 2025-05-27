"""
ML Prediction Module

Loads the trained fishing detection model and makes predictions on AIS features.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from models import FishingPredictions, PredictionPoint

logger = logging.getLogger(__name__)

class FishingPredictor:
    """Predicts fishing activity using the trained ML model."""
    
    def __init__(self, model_path: str = "models/fishing_classifier.joblib", 
                 metadata_path: str = "models/model_metadata.json"):
        """
        Initialize the fishing predictor.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file
        metadata_path : str
            Path to the model metadata file
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.is_loaded = False
        
        # Default configuration
        self.classification_threshold = 0.5
        self.confidence_thresholds = {
            'low': 0.6,
            'medium': 0.8,
            'high': 0.9
        }
        
        # Load model and metadata
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the trained model and metadata."""
        try:
            # Load the model
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
            
            # Load metadata if available
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Update configuration from metadata
                if 'classification_threshold' in self.metadata:
                    self.classification_threshold = self.metadata['classification_threshold']
                
                if 'confidence_thresholds' in self.metadata:
                    self.confidence_thresholds.update(self.metadata['confidence_thresholds'])
                
                logger.info(f"Loaded model metadata from {self.metadata_path}")
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def predict_fishing(self, features_df: pd.DataFrame, mmsi: str) -> Optional[FishingPredictions]:
        """
        Predict fishing activity for engineered features.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Engineered features from FeatureEngineer
        mmsi : str
            Vessel identifier
            
        Returns:
        --------
        FishingPredictions or None if prediction fails
        """
        try:
            if not self.is_loaded:
                logger.error("Model not loaded")
                return None
            
            if features_df.empty:
                logger.error("Empty features DataFrame")
                return None
            
            logger.info(f"Making predictions for {len(features_df)} data points")
            
            # Validate feature columns
            if not self._validate_features(features_df):
                return None
            
            # Get the features in the correct order
            model_features = self._get_model_features(features_df)
            
            # Make predictions
            probabilities = self.model.predict_proba(model_features)[:, 1]  # Probability of fishing
            binary_predictions = probabilities >= self.classification_threshold
            
            # Calculate confidence levels
            confidence_levels = self._calculate_confidence(probabilities)
            
            # Create prediction points (we need timestamps and coordinates)
            predictions = []
            
            # For now, we'll create synthetic timestamps and use the first coordinate
            # In the real pipeline, this should come from the original AIS data
            base_time = datetime.utcnow()
            
            for i, (prob, is_fishing, confidence) in enumerate(zip(probabilities, binary_predictions, confidence_levels)):
                # Create a prediction point
                # Note: In the real pipeline, lat/lon/timestamp should come from original AIS data
                pred_point = PredictionPoint(
                    timestamp=base_time,  # This should be the actual timestamp from AIS data
                    lat=0.0,  # These should be actual coordinates from AIS data
                    lon=0.0,
                    fishing_probability=float(prob),
                    is_fishing=bool(is_fishing),
                    confidence=confidence
                )
                predictions.append(pred_point)
            
            # Create FishingPredictions object
            model_version = self.metadata.get('model_version', '1.0') if self.metadata else '1.0'
            fishing_predictions = FishingPredictions(
                mmsi=mmsi,
                predictions=predictions,
                model_version=model_version,
                prediction_time=datetime.utcnow()
            )
            
            logger.info(f"Predicted {fishing_predictions.fishing_points_count} fishing points out of {fishing_predictions.total_points_count}")
            
            return fishing_predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None
    
    def predict_single_trip(self, features_df: pd.DataFrame, mmsi: str, 
                          timestamps: pd.Series, coordinates: pd.DataFrame) -> Optional[FishingPredictions]:
        """
        Predict fishing activity with proper timestamp and coordinate information.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Engineered features
        mmsi : str
            Vessel identifier
        timestamps : pd.Series
            Original timestamps from AIS data
        coordinates : pd.DataFrame
            Original coordinates (lat, lon) from AIS data
            
        Returns:
        --------
        FishingPredictions with proper metadata
        """
        try:
            if not self.is_loaded:
                logger.error("Model not loaded")
                return None
            
            # Validate inputs
            if len(features_df) != len(timestamps) or len(features_df) != len(coordinates):
                logger.error("Mismatched input lengths")
                return None
            
            # Get model features
            model_features = self._get_model_features(features_df)
            
            # Make predictions
            probabilities = self.model.predict_proba(model_features)[:, 1]
            binary_predictions = probabilities >= self.classification_threshold
            confidence_levels = self._calculate_confidence(probabilities)
            
            # Create prediction points with proper metadata
            predictions = []
            for i, (timestamp, lat, lon, prob, is_fishing, confidence) in enumerate(
                zip(timestamps, coordinates['lat'], coordinates['lon'], 
                    probabilities, binary_predictions, confidence_levels)):
                
                pred_point = PredictionPoint(
                    timestamp=pd.to_datetime(timestamp),
                    lat=float(lat),
                    lon=float(lon),
                    fishing_probability=float(prob),
                    is_fishing=bool(is_fishing),
                    confidence=confidence
                )
                predictions.append(pred_point)
            
            # Create FishingPredictions object
            model_version = self.metadata.get('model_version', '1.0') if self.metadata else '1.0'
            fishing_predictions = FishingPredictions(
                mmsi=mmsi,
                predictions=predictions,
                model_version=model_version,
                prediction_time=datetime.utcnow()
            )
            
            return fishing_predictions
            
        except Exception as e:
            logger.error(f"Single trip prediction failed: {str(e)}")
            return None
    
    def _validate_features(self, features_df: pd.DataFrame) -> bool:
        """Validate that features match model requirements."""
        if self.metadata and 'feature_names' in self.metadata:
            required_features = set(self.metadata['feature_names'])
            available_features = set(features_df.columns)
            
            missing_features = required_features - available_features
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return False
        
        # Check for NaN or infinite values
        if features_df.isna().any().any():
            logger.error("Features contain NaN values")
            return False
        
        if np.isinf(features_df.values).any():
            logger.error("Features contain infinite values")
            return False
        
        return True
    
    def _get_model_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Get features in the correct order for the model."""
        if self.metadata and 'feature_names' in self.metadata:
            # Use the exact feature order from training
            feature_names = self.metadata['feature_names']
            return features_df[feature_names]
        else:
            # Use all available features (fallback)
            return features_df
    
    def _calculate_confidence(self, probabilities: np.ndarray) -> list:
        """Calculate confidence levels based on prediction probabilities."""
        confidence_levels = []
        
        for prob in probabilities:
            # Distance from decision boundary (0.5)
            distance_from_boundary = abs(prob - 0.5)
            
            if distance_from_boundary >= (self.confidence_thresholds['high'] - 0.5):
                confidence = 'high'
            elif distance_from_boundary >= (self.confidence_thresholds['medium'] - 0.5):
                confidence = 'medium'
            else:
                confidence = 'low'
            
            confidence_levels.append(confidence)
        
        return confidence_levels
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        info = {
            'model_loaded': True,
            'model_path': self.model_path,
            'classification_threshold': self.classification_threshold,
            'confidence_thresholds': self.confidence_thresholds
        }
        
        if self.metadata:
            info.update({
                'model_version': self.metadata.get('model_version', 'unknown'),
                'training_accuracy': self.metadata.get('training_accuracy', 'unknown'),
                'feature_count': self.metadata.get('feature_count', 'unknown'),
                'training_date': self.metadata.get('training_date', 'unknown')
            })
        
        return info


def predict_fishing_activity(features_df: pd.DataFrame, mmsi: str, 
                           model_path: str = "models/fishing_classifier.joblib") -> Optional[FishingPredictions]:
    """
    Convenience function to predict fishing activity.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Engineered features
    mmsi : str
        Vessel identifier
    model_path : str
        Path to model file
        
    Returns:
    --------
    FishingPredictions or None if failed
    """
    predictor = FishingPredictor(model_path)
    return predictor.predict_fishing(features_df, mmsi)