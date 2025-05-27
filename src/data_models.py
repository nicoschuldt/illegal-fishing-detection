"""
Data models for the Illegal Fishing Detection Pipeline.

These models define the interfaces between pipeline components and ensure
type safety and data validation across the system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class AISPoint:
    """Single AIS data point from a vessel."""
    mmsi: str
    timestamp: datetime
    lat: float
    lon: float
    speed: float
    course: float
    distance_from_shore: float
    distance_from_port: float
    
    def __post_init__(self):
        """Validate AIS point data."""
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"Invalid latitude: {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"Invalid longitude: {self.lon}")
        if self.speed < 0:
            raise ValueError(f"Invalid speed: {self.speed}")
        if not (0 <= self.course <= 360):
            raise ValueError(f"Invalid course: {self.course}")


@dataclass
class AISData:
    """Collection of AIS points for a vessel."""
    mmsi: str
    points: List[AISPoint]
    start_time: datetime
    end_time: datetime
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'AISData':
        """Create AISData from pandas DataFrame."""
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        
        required_columns = [
            'mmsi', 'timestamp', 'lat', 'lon', 'speed', 
            'course', 'distance_from_shore', 'distance_from_port'
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        points = []
        for _, row in df.iterrows():
            points.append(AISPoint(
                mmsi=str(row['mmsi']),
                timestamp=pd.to_datetime(row['timestamp']),
                lat=float(row['lat']),
                lon=float(row['lon']),
                speed=float(row['speed']),
                course=float(row['course']),
                distance_from_shore=float(row['distance_from_shore']),
                distance_from_port=float(row['distance_from_port'])
            ))
        
        return cls(
            mmsi=str(df['mmsi'].iloc[0]),
            points=points,
            start_time=pd.to_datetime(df['timestamp'].min()),
            end_time=pd.to_datetime(df['timestamp'].max())
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert AISData to pandas DataFrame."""
        data = []
        for point in self.points:
            data.append({
                'mmsi': point.mmsi,
                'timestamp': point.timestamp,
                'lat': point.lat,
                'lon': point.lon,
                'speed': point.speed,
                'course': point.course,
                'distance_from_shore': point.distance_from_shore,
                'distance_from_port': point.distance_from_port
            })
        return pd.DataFrame(data)


@dataclass
class PredictionPoint:
    """Fishing prediction for a single AIS point."""
    timestamp: datetime
    lat: float
    lon: float
    fishing_probability: float
    is_fishing: bool
    confidence: str  # 'low', 'medium', 'high'
    
    def __post_init__(self):
        """Validate prediction data."""
        if not (0.0 <= self.fishing_probability <= 1.0):
            raise ValueError(f"Invalid probability: {self.fishing_probability}")
        if self.confidence not in ['low', 'medium', 'high']:
            raise ValueError(f"Invalid confidence: {self.confidence}")


@dataclass
class FishingPredictions:
    """Collection of fishing predictions for a vessel."""
    mmsi: str
    predictions: List[PredictionPoint]
    model_version: str
    prediction_time: datetime
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, mmsi: str, model_version: str) -> 'FishingPredictions':
        """Create FishingPredictions from pandas DataFrame."""
        required_columns = [
            'timestamp', 'lat', 'lon', 'fishing_probability', 
            'is_fishing', 'confidence'
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        predictions = []
        for _, row in df.iterrows():
            predictions.append(PredictionPoint(
                timestamp=pd.to_datetime(row['timestamp']),
                lat=float(row['lat']),
                lon=float(row['lon']),
                fishing_probability=float(row['fishing_probability']),
                is_fishing=bool(row['is_fishing']),
                confidence=str(row['confidence'])
            ))
        
        return cls(
            mmsi=mmsi,
            predictions=predictions,
            model_version=model_version,
            prediction_time=datetime.utcnow()
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert FishingPredictions to pandas DataFrame."""
        data = []
        for pred in self.predictions:
            data.append({
                'timestamp': pred.timestamp,
                'lat': pred.lat,
                'lon': pred.lon,
                'fishing_probability': pred.fishing_probability,
                'is_fishing': pred.is_fishing,
                'confidence': pred.confidence
            })
        return pd.DataFrame(data)
    
    @property
    def fishing_points_count(self) -> int:
        """Count of points classified as fishing."""
        return sum(1 for p in self.predictions if p.is_fishing)
    
    @property
    def total_points_count(self) -> int:
        """Total number of prediction points."""
        return len(self.predictions)
    
    @property
    def fishing_percentage(self) -> float:
        """Percentage of points classified as fishing."""
        if self.total_points_count == 0:
            return 0.0
        return (self.fishing_points_count / self.total_points_count) * 100


@dataclass
class MPAViolation:
    """Single MPA violation event."""
    mpa_name: str
    entry_time: datetime
    exit_time: Optional[datetime]
    fishing_points: int
    severity: str  # 'low', 'medium', 'high'
    mpa_id: Optional[str] = None
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Duration of violation in minutes."""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return None


@dataclass
class MPAAnalysis:
    """Results of MPA violation analysis."""
    mmsi: str
    analysis_time: datetime
    total_fishing_points: int
    fishing_in_mpa_points: int
    violation_detected: bool
    violations: List[MPAViolation]
    risk_score: float  # 0.0 to 1.0
    
    @property
    def violation_percentage(self) -> float:
        """Percentage of fishing points that are in MPAs."""
        if self.total_fishing_points == 0:
            return 0.0
        return (self.fishing_in_mpa_points / self.total_fishing_points) * 100


@dataclass
class FishingSummary:
    """Summary of fishing activity for a vessel."""
    total_ais_points: int
    fishing_points: int
    fishing_percentage: float
    avg_fishing_probability: float
    time_period_hours: float


@dataclass
class AnalysisResult:
    """Final result of illegal fishing analysis."""
    mmsi: str
    analysis_time: datetime
    success: bool
    time_period_start: Optional[datetime] = None
    time_period_end: Optional[datetime] = None
    fishing_summary: Optional[FishingSummary] = None
    mpa_analysis: Optional[MPAAnalysis] = None
    violation_detected: bool = False
    risk_score: float = 0.0
    risk_level: str = "LOW"  # 'LOW', 'MEDIUM', 'HIGH'
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'mmsi': self.mmsi,
            'analysis_time': self.analysis_time.isoformat(),
            'success': self.success,
            'violation_detected': self.violation_detected,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level
        }
        
        if self.error_message:
            result['error'] = self.error_message
        
        if self.time_period_start and self.time_period_end:
            result['time_period'] = {
                'start': self.time_period_start.isoformat(),
                'end': self.time_period_end.isoformat(),
                'duration_hours': (self.time_period_end - self.time_period_start).total_seconds() / 3600
            }
        
        if self.fishing_summary:
            result['fishing_summary'] = {
                'total_ais_points': self.fishing_summary.total_ais_points,
                'fishing_points': self.fishing_summary.fishing_points,
                'fishing_percentage': self.fishing_summary.fishing_percentage,
                'avg_fishing_probability': self.fishing_summary.avg_fishing_probability,
                'time_period_hours': self.fishing_summary.time_period_hours
            }
        
        if self.mpa_analysis:
            result['mpa_analysis'] = {
                'total_fishing_points': self.mpa_analysis.total_fishing_points,
                'fishing_in_mpa_points': self.mpa_analysis.fishing_in_mpa_points,
                'violation_percentage': self.mpa_analysis.violation_percentage,
                'violations': [
                    {
                        'mpa_name': v.mpa_name,
                        'entry_time': v.entry_time.isoformat(),
                        'exit_time': v.exit_time.isoformat() if v.exit_time else None,
                        'fishing_points': v.fishing_points,
                        'severity': v.severity,
                        'duration_minutes': v.duration_minutes
                    }
                    for v in self.mpa_analysis.violations
                ]
            }
        
        return result


# Configuration Models
@dataclass
class ModelConfig:
    """Configuration for ML model."""
    model_path: str
    feature_names_path: str
    classification_threshold: float = 0.5
    confidence_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = {
                'low': 0.6,
                'medium': 0.8,
                'high': 0.9
            }


@dataclass
class APIConfig:
    """Configuration for external API calls."""
    gfw_api_base: str
    rate_limit: int = 60  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    api_key: Optional[str] = None


# Utility functions for data validation
def validate_mmsi(mmsi: str) -> bool:
    """Validate MMSI format."""
    return len(mmsi) == 9 and mmsi.isdigit()


def validate_time_window(hours_back: int) -> bool:
    """Validate time window is reasonable."""
    return 1 <= hours_back <= 168  # 1 hour to 1 week


def create_error_result(mmsi: str, error_message: str) -> AnalysisResult:
    """Create standardized error result."""
    return AnalysisResult(
        mmsi=mmsi,
        analysis_time=datetime.utcnow(),
        success=False,
        error_message=error_message
    )