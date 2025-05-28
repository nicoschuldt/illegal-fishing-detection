"""
Illegal Fishing Detection Pipeline

Main orchestration module that coordinates all components for detecting
illegal fishing activity from vessel AIS data.
"""

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import numpy as np
from pathlib import Path

# Import our modules
from models import (
    AISData, FishingPredictions, AnalysisResult, FishingSummary, MPAAnalysis, MPAViolation,
    create_error_result, validate_mmsi, validate_time_window
)
from src.features.preprocessor import AISPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.predictor import FishingPredictor

# Import MPA components
try:
    from src.features.mpa.loader import MPALoader
    from src.features.mpa.utils import MPAQuery
    from src.config import WDPA_GDB, WDPA_GDB_LAYER, WDPA_CSV
    MPA_AVAILABLE = True
except ImportError as e:
    print(f"MPA components not available: {e}")
    MPA_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IllegalFishingDetector:
    """
    Main pipeline class for detecting illegal fishing activity.
    
    This class coordinates all components of the pipeline:
    1. AIS data preprocessing
    2. Feature engineering
    3. Fishing activity prediction
    4. MPA violation analysis
    """
    
    def __init__(self, model_path: str = "models/fishing_classifier.joblib",
                 mpa_gdb_path: str = None, mpa_layer: str = None, mpa_csv_path: str = None):
        """
        Initialize the illegal fishing detector.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained ML model
        mpa_gdb_path : str
            Path to MPA GDB file (optional)
        mpa_layer : str  
            MPA layer name (optional)
        mpa_csv_path : str
            Path to MPA CSV file (optional)
        """
        try:
            self.preprocessor = AISPreprocessor()
            self.feature_engineer = FeatureEngineer()
            self.predictor = FishingPredictor(model_path)
            
            # Initialize MPA components if available
            self.mpa_loader = None
            self.mpa_query = None
            
            if MPA_AVAILABLE and mpa_gdb_path and mpa_layer and mpa_csv_path:
                try:
                    self.mpa_loader = MPALoader(
                        gdb_path=mpa_gdb_path,
                        layer_name=mpa_layer,
                        csv_path=mpa_csv_path
                    )
                    self.mpa_query = MPAQuery(self.mpa_loader)
                    logger.info(f"MPA analysis enabled with {len(self.mpa_loader.gdf)} MPAs")
                except Exception as e:
                    logger.warning(f"Failed to initialize MPA components: {str(e)}")
            else:
                logger.info("MPA analysis disabled - using mock analysis")
            
            logger.info("Illegal Fishing Detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize detector: {str(e)}")
            raise
    
    def analyze_vessel_data(self, ais_dataframe: pd.DataFrame, mmsi: str) -> AnalysisResult:
        """
        Complete analysis of vessel AIS data including MPA violation detection.
        
        Parameters:
        -----------
        ais_dataframe : pd.DataFrame
            Raw AIS data from the AIS fetcher component
        mmsi : str
            Vessel identifier
            
        Returns:
        --------
        AnalysisResult with fishing predictions and MPA analysis
        """
        try:
            logger.info(f"Starting complete analysis for vessel {mmsi}")
            
            # Validate inputs (relaxed validation for anonymized data)
            if not mmsi or len(str(mmsi)) < 5:  # Relaxed MMSI validation for testing
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
            
            # Step 4: MPA Analysis
            logger.info("Analyzing MPA violations...")
            mpa_analysis = self.analyze_mpa_violations(predictions)
            
            if mpa_analysis is None:
                logger.warning("MPA analysis failed, continuing without MPA data")
            
            # Step 5: Create comprehensive analysis result
            result = self._create_complete_analysis_result(ais_data, predictions, mpa_analysis)
            
            logger.info(f"Complete analysis finished for vessel {mmsi}")
            return result
            
        except Exception as e:
            logger.error(f"Complete pipeline error for vessel {mmsi}: {str(e)}")
            return create_error_result(mmsi, f"Pipeline error: {str(e)}")
    
    def analyze_mpa_violations(self, predictions: FishingPredictions) -> Optional[MPAAnalysis]:
        """
        Analyze fishing predictions against Marine Protected Areas.
        
        Parameters:
        -----------
        predictions : FishingPredictions
            Fishing predictions with coordinates
            
        Returns:
        --------
        MPAAnalysis or None if analysis fails
        """
        try:
            if not self.mpa_loader or not self.mpa_query:
                # Mock MPA analysis for testing
                return self._create_mock_mpa_analysis(predictions)
            
            # Convert predictions to GeoDataFrame
            ais_data = []
            for pred in predictions.predictions:
                ais_data.append({
                    'timestamp': pred.timestamp,
                    'lat': pred.lat,
                    'lon': pred.lon,
                    'fishing_probability': pred.fishing_probability,
                    'is_fishing': pred.is_fishing,
                    'confidence': pred.confidence
                })
            
            ais_df = pd.DataFrame(ais_data)
            gdf_ais = gpd.GeoDataFrame(
                ais_df,
                geometry=gpd.points_from_xy(ais_df.lon, ais_df.lat),
                crs="EPSG:4326"
            )
            
            # Spatial join with MPAs (using Oscar's approach)
            mpa_gdf = self.mpa_loader.gdf
            gdf_enriched = gpd.sjoin(
                gdf_ais,
                mpa_gdf[["WDPAID", "PA_DEF", "NO_TAKE", "geometry"]],
                how="left",
                predicate="within"
            ).drop(columns="index_right")
            
            # Add distance to nearest MPA
            gdf_enriched["dist_to_mpa"] = gdf_enriched.geometry.apply(
                lambda p: self.mpa_query.distance_to_nearest_mpa(p.y, p.x)
            )
            
            # Analyze violations
            violations = self._identify_violations(gdf_enriched)
            
            # Calculate summary statistics
            total_fishing_points = gdf_enriched['is_fishing'].sum()
            fishing_in_mpa_points = gdf_enriched[
                (gdf_enriched['is_fishing'] == True) & 
                (gdf_enriched['WDPAID'].notna())
            ].shape[0]
            
            violation_detected = len(violations) > 0
            risk_score = self._calculate_mpa_risk_score(gdf_enriched)
            
            mpa_analysis = MPAAnalysis(
                mmsi=predictions.mmsi,
                analysis_time=datetime.utcnow(),
                total_fishing_points=int(total_fishing_points),
                fishing_in_mpa_points=int(fishing_in_mpa_points),
                violation_detected=violation_detected,
                violations=violations,
                risk_score=risk_score
            )
            
            return mpa_analysis
            
        except Exception as e:
            logger.error(f"MPA analysis failed: {str(e)}")
            return None
    
    def _identify_violations(self, gdf_enriched: gpd.GeoDataFrame) -> List[MPAViolation]:
        """Identify specific MPA violations from enriched AIS data."""
        violations = []
        
        # Find fishing points within MPAs
        violation_points = gdf_enriched[
            (gdf_enriched['is_fishing'] == True) & 
            (gdf_enriched['WDPAID'].notna())
        ]
        
        if len(violation_points) == 0:
            return violations
        
        # Group by MPA to create violation events
        for wdpa_id, group in violation_points.groupby('WDPAID'):
            if len(group) >= 3:  # Require at least 3 points for a violation
                # Determine severity based on number of points and no-take status
                no_take = group['NO_TAKE'].iloc[0] if 'NO_TAKE' in group.columns else 'Unknown'
                fishing_points = len(group)
                
                if no_take == 'All' and fishing_points >= 10:
                    severity = 'high'
                elif no_take == 'All' and fishing_points >= 5:
                    severity = 'medium'
                elif fishing_points >= 10:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                mpa_name = group['PA_DEF'].iloc[0] if 'PA_DEF' in group.columns else f'MPA_{wdpa_id}'
                
                violation = MPAViolation(
                    mpa_name=mpa_name,
                    entry_time=group['timestamp'].min(),
                    exit_time=group['timestamp'].max(),
                    fishing_points=fishing_points,
                    severity=severity,
                    mpa_id=str(wdpa_id)
                )
                
                violations.append(violation)
        
        return violations
    
    def _calculate_mpa_risk_score(self, gdf_enriched: gpd.GeoDataFrame) -> float:
        """Calculate overall MPA violation risk score."""
        total_fishing = gdf_enriched['is_fishing'].sum()
        
        if total_fishing == 0:
            return 0.0
        
        fishing_in_mpa = gdf_enriched[
            (gdf_enriched['is_fishing'] == True) & 
            (gdf_enriched['WDPAID'].notna())
        ].shape[0]
        
        # Base risk from fishing in MPAs
        base_risk = fishing_in_mpa / total_fishing
        
        # Bonus for no-take areas
        no_take_fishing = gdf_enriched[
            (gdf_enriched['is_fishing'] == True) & 
            (gdf_enriched['WDPAID'].notna()) &
            (gdf_enriched['NO_TAKE'] == 'All')
        ].shape[0] if 'NO_TAKE' in gdf_enriched.columns else 0
        
        no_take_bonus = (no_take_fishing / total_fishing) * 0.3 if total_fishing > 0 else 0
        
        return min(1.0, base_risk + no_take_bonus)
    
    def _create_mock_mpa_analysis(self, predictions: FishingPredictions) -> MPAAnalysis:
        """Create mock MPA analysis for testing when MPA data is not available."""
        total_fishing_points = predictions.fishing_points_count
        
        # Simulate some violations for demonstration
        mock_violations = []
        if total_fishing_points > 10:
            mock_violations.append(MPAViolation(
                mpa_name="Demo Marine Reserve",
                entry_time=predictions.predictions[0].timestamp,
                exit_time=predictions.predictions[-1].timestamp,
                fishing_points=min(5, total_fishing_points // 2),
                severity='medium',
                mpa_id='DEMO_001'
            ))
        
        return MPAAnalysis(
            mmsi=predictions.mmsi,
            analysis_time=datetime.utcnow(),
            total_fishing_points=total_fishing_points,
            fishing_in_mpa_points=len(mock_violations) * 5 if mock_violations else 0,
            violation_detected=len(mock_violations) > 0,
            violations=mock_violations,
            risk_score=0.3 if mock_violations else 0.0
        )
    
    def _create_complete_analysis_result(self, ais_data: AISData, predictions: FishingPredictions, 
                                       mpa_analysis: Optional[MPAAnalysis]) -> AnalysisResult:
        """Create complete analysis result including MPA analysis."""
        
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
        
        # Calculate overall risk score
        base_risk = avg_fishing_prob
        mpa_risk_bonus = 0.0
        
        if mpa_analysis and mpa_analysis.violation_detected:
            mpa_risk_bonus = mpa_analysis.risk_score * 0.4  # 40% bonus for MPA violations
        
        total_risk_score = min(1.0, base_risk + mpa_risk_bonus)
        
        # Create complete analysis result
        result = AnalysisResult(
            mmsi=predictions.mmsi,
            analysis_time=datetime.utcnow(),
            success=True,
            time_period_start=ais_data.start_time,
            time_period_end=ais_data.end_time,
            fishing_summary=fishing_summary,
            mpa_analysis=mpa_analysis,
            violation_detected=mpa_analysis.violation_detected if mpa_analysis else False,
            risk_score=total_risk_score,
            risk_level=self._calculate_risk_level(total_risk_score)
        )
        
        return result
    
    def _calculate_risk_level(self, risk_score: float) -> str:
        """Calculate risk level from risk score."""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def create_analysis_map(self, predictions: FishingPredictions, mpa_analysis: Optional[MPAAnalysis] = None, 
                          output_file: str = "fishing_analysis_map.html") -> str:
        """
        Create an interactive map showing fishing analysis results.
        
        Parameters:
        -----------
        predictions : FishingPredictions
            Fishing predictions with coordinates
        mpa_analysis : MPAAnalysis, optional
            MPA violation analysis results
        output_file : str
            Output HTML file name
            
        Returns:
        --------
        str: Path to the created map file
        """
        try:
            # Calculate map center
            lats = [p.lat for p in predictions.predictions]
            lons = [p.lon for p in predictions.predictions]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles='OpenStreetMap'
            )
            
            # Add fishing/non-fishing points
            fishing_coords = []
            non_fishing_coords = []
            
            for pred in predictions.predictions:
                coord = [pred.lat, pred.lon]
                if pred.is_fishing:
                    fishing_coords.append(coord)
                    # Add fishing point marker
                    folium.CircleMarker(
                        location=coord,
                        radius=4,
                        popup=f"Fishing: {pred.fishing_probability:.2f}",
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.7
                    ).add_to(m)
                else:
                    non_fishing_coords.append(coord)
                    # Add non-fishing point marker (less prominent)
                    if len(non_fishing_coords) % 5 == 0:  # Show every 5th point to avoid clutter
                        folium.CircleMarker(
                            location=coord,
                            radius=2,
                            popup=f"Transit: {pred.fishing_probability:.2f}",
                            color='blue',
                            fill=True,
                            fillColor='blue',
                            fillOpacity=0.5
                        ).add_to(m)
            
            # Add trajectory line
            if len(lats) > 1:
                trajectory_coords = [[lat, lon] for lat, lon in zip(lats, lons)]
                folium.PolyLine(
                    locations=trajectory_coords,
                    color='gray',
                    weight=2,
                    opacity=0.8
                ).add_to(m)
            
            # Add MPA violations if available
            if mpa_analysis and mpa_analysis.violations:
                for violation in mpa_analysis.violations:
                    # Add violation marker
                    folium.Marker(
                        location=[lats[0], lons[0]],  # Approximate location
                        popup=f"MPA Violation: {violation.mpa_name}<br>"
                              f"Severity: {violation.severity}<br>"
                              f"Fishing points: {violation.fishing_points}",
                        icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
                    ).add_to(m)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <b>Fishing Activity</b><br>
            <i class="fa fa-circle" style="color:red"></i> Fishing Activity<br>
            <i class="fa fa-circle" style="color:blue"></i> Transit/Non-fishing<br>
            <i class="fa fa-exclamation-triangle" style="color:red"></i> MPA Violation<br>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add title
            title_html = f'''
            <h3 align="center" style="font-size:20px"><b>Vessel {predictions.mmsi} - Fishing Analysis</b></h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Save map
            m.save(output_file)
            logger.info(f"Analysis map saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to create analysis map: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = self.predictor.get_model_info()
        info.update({
            'mpa_analysis_enabled': self.mpa_loader is not None,
            'mpa_count': len(self.mpa_loader.gdf) if self.mpa_loader else 0
        })
        return info


# Comprehensive testing with real CSV data
def test_complete_pipeline():
    """Test the complete pipeline with real CSV data and create visualizations."""
    
    print("="*80)
    print("TESTING COMPLETE ILLEGAL FISHING DETECTION PIPELINE")
    print("="*80)
    
    # Load real CSV data
    csv_path = "data/raw/drifting_longlines.csv"
    
    try:
        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} AIS points from CSV")
        
        # Get unique vessels
        vessels = df['mmsi'].unique()
        print(f"Found {len(vessels)} unique vessels")
        
        # Test with first few vessels
        test_vessels = vessels[:3]  # Test with first 3 vessels
        
        # Initialize detector
        detector = IllegalFishingDetector()
        
        print(f"\nTesting with vessels: {test_vessels}")
        
        results = []
        
        for mmsi in test_vessels:
            # Convert MMSI to string and handle scientific notation
            mmsi_str = f"{int(mmsi)}" if isinstance(mmsi, (int, float)) else str(mmsi)
            
            print(f"\n{'='*50}")
            print(f"ANALYZING VESSEL: {mmsi_str}")
            print(f"{'='*50}")
            
            # Get data for this vessel
            vessel_data = df[df['mmsi'] == mmsi].copy()
            print(f"Vessel has {len(vessel_data)} AIS points")
            
            if len(vessel_data) < 10:
                print("Skipping vessel with insufficient data")
                continue
            
            # Sample data if too large (for testing) - preserve chronological order
            if len(vessel_data) > 500:
                vessel_data = vessel_data.sort_values('timestamp').tail(500)
                print(f"Sampled to {len(vessel_data)} points for testing (most recent data)")
            
            # Analyze vessel
            result = detector.analyze_vessel_data(vessel_data, mmsi_str)
            
            if result.success:
                print(f"\n✅ Analysis successful for vessel {mmsi}")
                print(f"   Time period: {result.time_period_start} to {result.time_period_end}")
                print(f"   Total AIS points: {result.fishing_summary.total_ais_points}")
                print(f"   Fishing points: {result.fishing_summary.fishing_points}")
                print(f"   Fishing percentage: {result.fishing_summary.fishing_percentage:.1f}%")
                print(f"   Risk level: {result.risk_level}")
                print(f"   Risk score: {result.risk_score:.3f}")
                
                if result.mpa_analysis:
                    print(f"   MPA violations: {len(result.mpa_analysis.violations)}")
                    if result.violation_detected:
                        print(f"   🚨 ILLEGAL FISHING DETECTED!")
                        for violation in result.mpa_analysis.violations:
                            print(f"      - {violation.mpa_name}: {violation.fishing_points} fishing points ({violation.severity} severity)")
                
                # Create visualization for this vessel
                # First get the predictions (we need to run the analysis again to get predictions object)
                print(f"   Creating visualization map...")
                
                # Re-run analysis to get predictions object
                ais_data = detector.preprocessor.preprocess(vessel_data)
                if ais_data:
                    features = detector.feature_engineer.engineer_features(ais_data)
                    if features is not None:
                        timestamps = pd.Series([p.timestamp for p in ais_data.points])
                        coordinates = pd.DataFrame({
                            'lat': [p.lat for p in ais_data.points],
                            'lon': [p.lon for p in ais_data.points]
                        })
                        predictions = detector.predictor.predict_single_trip(
                            features, mmsi_str, timestamps, coordinates
                        )
                        
                        if predictions:
                            map_file = f"vessel_{mmsi_str}_analysis_map.html"
                            detector.create_analysis_map(predictions, result.mpa_analysis, map_file)
                            print(f"   📍 Map saved: {map_file}")
                
                results.append(result)
                
            else:
                print(f"❌ Analysis failed for vessel {mmsi_str}: {result.error_message}")
        
        # Summary
        print(f"\n{'='*80}")
        print("PIPELINE TEST SUMMARY")
        print(f"{'='*80}")
        
        if results:
            successful_analyses = len(results)
            total_violations = sum(1 for r in results if r.violation_detected)
            avg_fishing_percentage = sum(r.fishing_summary.fishing_percentage for r in results) / len(results)
            
            print(f"✅ Successful analyses: {successful_analyses}")
            print(f"🎣 Average fishing activity: {avg_fishing_percentage:.1f}%")
            print(f"🚨 Vessels with violations: {total_violations}")
            
            print(f"\nDetailed results:")
            for result in results:
                status = "🚨 VIOLATION" if result.violation_detected else "✅ Clean"
                print(f"   Vessel {result.mmsi}: {result.fishing_summary.fishing_percentage:.1f}% fishing, "
                      f"Risk: {result.risk_level} ({result.risk_score:.3f}) - {status}")
        
        else:
            print("❌ No successful analyses completed")
        
        print(f"\n🎉 Pipeline test completed!")
        return results
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file: {csv_path}")
        print("   Please ensure the CSV file exists in the specified location")
        return None
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def analyze_single_vessel(mmsi: str, hours_back: int = 24):
    """
    Analyze a single vessel using data from the CSV file.
    
    Parameters:
    -----------
    mmsi : str
        Vessel MMSI to analyze
    hours_back : int
        Hours of data to use (most recent)
    """
    csv_path = "data/raw/drifting_longlines.csv"
    
    try:
        print(f"Analyzing vessel {mmsi}...")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Handle MMSI conversion for lookup
        try:
            mmsi_numeric = int(float(mmsi))
            vessel_data = df[df['mmsi'] == mmsi_numeric].copy()
        except:
            vessel_data = df[df['mmsi'].astype(str) == mmsi].copy()
        
        if vessel_data.empty:
            print(f"No data found for vessel {mmsi}")
            return None
        
        print(f"Found {len(vessel_data)} data points for vessel {mmsi}")
        
        # Use most recent data
        vessel_data = vessel_data.sort_values('timestamp').tail(hours_back * 4)  # Assuming ~4 points per hour
        
        # Analyze
        detector = IllegalFishingDetector()
        result = detector.analyze_vessel_data(vessel_data, mmsi)
        
        if result.success:
            print(f"\n🎣 Fishing Analysis Results for Vessel {mmsi}")
            print(f"   Fishing percentage: {result.fishing_summary.fishing_percentage:.1f}%")
            print(f"   Risk level: {result.risk_level}")
            print(f"   Violations detected: {result.violation_detected}")
            
            # Create map
            ais_data = detector.preprocessor.preprocess(vessel_data)
            if ais_data:
                features = detector.feature_engineer.engineer_features(ais_data)
                if features is not None:
                    timestamps = pd.Series([p.timestamp for p in ais_data.points])
                    coordinates = pd.DataFrame({
                        'lat': [p.lat for p in ais_data.points],
                        'lon': [p.lon for p in ais_data.points]
                    })
                    predictions = detector.predictor.predict_single_trip(
                        features, mmsi, timestamps, coordinates
                    )
                    
                    if predictions:
                        map_file = f"vessel_{mmsi}_detailed_analysis.html"
                        detector.create_analysis_map(predictions, result.mpa_analysis, map_file)
                        print(f"   📍 Detailed map: {map_file}")
        
        return result
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Run comprehensive pipeline test
    print("Starting comprehensive pipeline test with real data...")
    test_complete_pipeline()
    
    # Optionally analyze a specific vessel
    # Uncomment and modify MMSI to test a specific vessel
    # analyze_single_vessel("123456789", hours_back=12)