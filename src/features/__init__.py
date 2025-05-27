"""
Features package for illegal fishing detection.

This package contains modules for preprocessing AIS data and engineering
features for the fishing detection model.
"""

from .preprocessor import AISPreprocessor, preprocess_ais_dataframe
from .trip_segmenter import TripSegmenter, segment_ais_data
from .feature_engineer import FeatureEngineer, engineer_ais_features, TripFeatureEngineer
from .mpa import MPALoader, MPAQuery
from data_models import AISData, FishingPredictions, AnalysisResult, FishingSummary, MPAViolation, MPAAnalysis

__all__ = [
    'AISPreprocessor',
    'preprocess_ais_dataframe',
    'TripSegmenter', 
    'segment_ais_data',
    'FeatureEngineer',
    'engineer_ais_features',
    'TripFeatureEngineer',
    'MPALoader',
    'MPAQuery',
    'AISData',
    'FishingPredictions',
    'AnalysisResult',
    'FishingSummary',
    'MPAViolation',
    'MPAAnalysis'
]