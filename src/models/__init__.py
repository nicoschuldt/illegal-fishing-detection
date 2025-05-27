
"""
Models package for illegal fishing detection.

This package contains ML model components for predicting fishing activity.
"""

from .predictor import FishingPredictor, predict_fishing_activity

__all__ = [
    'FishingPredictor',
    'predict_fishing_activity'
]