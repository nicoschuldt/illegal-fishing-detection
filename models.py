"""
Data models for the Illegal Fishing Detection Pipeline.

This module re-exports all data models from src.data_models for easier importing.
"""

# Import all data models from the src package
from src.data_models import (
    AISPoint,
    AISData,
    PredictionPoint,
    FishingPredictions,
    MPAViolation,
    MPAAnalysis,
    FishingSummary,
    AnalysisResult,
    ModelConfig,
    APIConfig,
    validate_mmsi,
    validate_time_window,
    create_error_result
)

# Re-export everything
__all__ = [
    'AISPoint',
    'AISData',
    'PredictionPoint',
    'FishingPredictions',
    'MPAViolation',
    'MPAAnalysis',
    'FishingSummary',
    'AnalysisResult',
    'ModelConfig',
    'APIConfig',
    'validate_mmsi',
    'validate_time_window',
    'create_error_result'
]
