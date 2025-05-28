"""
Illegal Fishing Detection Dashboard

A Streamlit web application for monitoring vessel fishing activity and detecting
illegal fishing in Marine Protected Areas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from streamlit_folium import st_folium
import logging
import math

# Import our pipeline components
from pipeline import IllegalFishingDetector
from models import AnalysisResult

# Configure page
st.set_page_config(
    page_title="Ocean Monitoring Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging to avoid cluttering the app
logging.getLogger().setLevel(logging.WARNING)

# Custom CSS for modern dashboard look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .violation-alert {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #d32f2f;
    }
    .clean-alert {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #2e7d32;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
    """Streamlit dashboard for illegal fishing detection."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.detector = None
        self.results = []
        
    def initialize_detector(self):
        """Initialize the fishing detector with caching."""
        if self.detector is None:
            with st.spinner("Initializing ML model..."):
                try:
                    self.detector = IllegalFishingDetector()
                    st.success("✅ ML model loaded successfully")
                except Exception as e:
                    st.error(f"❌ Failed to load ML model: {str(e)}")
                    return False
        return True
    
    def parse_mmsi_list(self, mmsi_input: str) -> list:
        """Parse comma-separated MMSI list."""
        if not mmsi_input.strip():
            return []
        
        mmsi_list = []
        for mmsi in mmsi_input.split(','):
            mmsi = mmsi.strip()
            if mmsi:
                mmsi_list.append(mmsi)
        
        return mmsi_list
    
    def load_vessel_data(self, mmsi: str, max_points: int = 1000) -> pd.DataFrame:
        """Load vessel data from CSV file."""
        try:
            df = pd.read_csv('data/raw/drifting_longlines.csv')
            
            # Handle MMSI conversion
            try:
                mmsi_numeric = int(float(mmsi))
                vessel_data = df[df['mmsi'] == mmsi_numeric].copy()
            except:
                vessel_data = df[df['mmsi'].astype(str) == mmsi].copy()
            
            if vessel_data.empty:
                return None
            
            # Sort by timestamp first to ensure chronological order
            vessel_data = vessel_data.sort_values('timestamp').reset_index(drop=True)
            
            # Sample data if too large - preserve chronological order
            if len(vessel_data) > max_points:
                # Option 1: Take most recent data (preserves continuous trajectory)
                vessel_data = vessel_data.tail(max_points)
                
                # Option 2: Systematic sampling (preserves overall trajectory shape)
                # step = len(vessel_data) // max_points
                # vessel_data = vessel_data.iloc[::step][:max_points]
            
            return vessel_data
            
        except Exception as e:
            st.error(f"Error loading data for vessel {mmsi}: {str(e)}")
            return None
    
    def analyze_vessels(self, mmsi_list: list) -> list:
        """Analyze multiple vessels."""
        results = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, mmsi in enumerate(mmsi_list):
            status_text.text(f"Analyzing vessel {mmsi}... ({i+1}/{len(mmsi_list)})")
            progress_bar.progress((i + 1) / len(mmsi_list))
            
            # Load vessel data
            vessel_data = self.load_vessel_data(mmsi)
            
            if vessel_data is None:
                results.append({
                    'mmsi': mmsi,
                    'status': 'error',
                    'message': 'No data found',
                    'result': None
                })
                continue
            
            # Analyze vessel
            try:
                result = self.detector.analyze_vessel_data(vessel_data, mmsi)
                
                if result.success:
                    results.append({
                        'mmsi': mmsi,
                        'status': 'success',
                        'message': 'Analysis completed',
                        'result': result,
                        'data_points': len(vessel_data),
                        'vessel_data': vessel_data  # Store vessel data for map creation
                    })
                else:
                    results.append({
                        'mmsi': mmsi,
                        'status': 'error',
                        'message': result.error_message,
                        'result': None
                    })
                    
            except Exception as e:
                results.append({
                    'mmsi': mmsi,
                    'status': 'error',
                    'message': str(e),
                    'result': None
                })
        
        status_text.text("Analysis completed!")
        progress_bar.empty()
        
        return results
    
    def create_combined_map(self, results: list) -> folium.Map:
        """Create a combined map showing all analyzed vessels."""
        
        # Calculate map center from all vessel positions
        all_lats, all_lons = [], []
        
        for item in results:
            if item['status'] == 'success' and item['result'] and 'vessel_data' in item:
                vessel_data = item['vessel_data']
                if vessel_data is not None and not vessel_data.empty:
                    # Validate coordinates
                    valid_coords = vessel_data[
                        (vessel_data['lat'].between(-90, 90)) & 
                        (vessel_data['lon'].between(-180, 180)) &
                        (vessel_data['lat'].notna()) &
                        (vessel_data['lon'].notna())
                    ]
                    if not valid_coords.empty:
                        all_lats.extend(valid_coords['lat'].tolist())
                        all_lons.extend(valid_coords['lon'].tolist())
        
        if not all_lats:
            # Default to global view if no data
            center_lat, center_lon = 0, 0
            zoom = 2
            st.warning("No valid coordinate data found for map display. Showing global view.")
        else:
            center_lat = sum(all_lats) / len(all_lats)
            center_lon = sum(all_lons) / len(all_lons)
            
            # Calculate appropriate zoom level based on coordinate spread
            lat_range = max(all_lats) - min(all_lats)
            lon_range = max(all_lons) - min(all_lons)
            max_range = max(lat_range, lon_range)
            
            if max_range > 50:
                zoom = 3
            elif max_range > 20:
                zoom = 4
            elif max_range > 10:
                zoom = 5
            elif max_range > 5:
                zoom = 6
            else:
                zoom = 7
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Color scheme for different vessel states
        colors = {
            'clean': '#4CAF50',      # Green - no violations
            'fishing': '#FF9800',     # Orange - fishing but legal
            'violation': '#F44336'    # Red - illegal fishing
        }
        
        # Track if any vessels were added to the map
        vessels_added = 0
        
        # Add vessels to map
        for item in results:
            if item['status'] != 'success' or not item['result'] or 'vessel_data' not in item:
                continue
                
            result = item['result']
            mmsi = item['mmsi']
            vessel_data = item['vessel_data']
            
            if vessel_data is None or vessel_data.empty:
                continue
            
            # Validate and clean coordinates
            valid_coords = vessel_data[
                (vessel_data['lat'].between(-90, 90)) & 
                (vessel_data['lon'].between(-180, 180)) &
                (vessel_data['lat'].notna()) &
                (vessel_data['lon'].notna())
            ]
            
            if valid_coords.empty:
                st.warning(f"No valid coordinates found for vessel {mmsi}")
                continue
            
            # Determine vessel status
            if result.violation_detected:
                status = 'violation'
                status_text = "🚨 ILLEGAL FISHING"
            elif result.fishing_summary.fishing_percentage > 30:
                status = 'fishing'
                status_text = "🎣 Fishing Activity"
            else:
                status = 'clean'
                status_text = "✅ Clean"
            
            # Create trajectory from valid coordinates (ensure chronological order)
            valid_coords = valid_coords.sort_values('timestamp').reset_index(drop=True)
            trajectory = [[row['lat'], row['lon']] for _, row in valid_coords.iterrows()]
            
            if len(trajectory) > 1:
                # Add trajectory line
                folium.PolyLine(
                    locations=trajectory,
                    color=colors[status],
                    weight=3,
                    opacity=0.8,
                    popup=f"Vessel {mmsi} - {status_text}"
                ).add_to(m)
                
                # Add start marker (green circle)
                folium.CircleMarker(
                    location=trajectory[0],
                    radius=6,
                    popup=f"Start: {mmsi}",
                    color='darkgreen',
                    fill=True,
                    fillColor='lightgreen',
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
                
                # Add end marker (red circle) 
                folium.CircleMarker(
                    location=trajectory[-1],
                    radius=6,
                    popup=f"End: {mmsi}",
                    color='darkred',
                    fill=True,
                    fillColor='lightcoral',
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
                
                # Add direction arrows along the path (every 10th point)
                if len(trajectory) > 10:
                    arrow_interval = max(1, len(trajectory) // 10)
                    for i in range(arrow_interval, len(trajectory), arrow_interval):
                        if i < len(trajectory) - 1:
                            # Calculate bearing for arrow direction
                            lat1, lon1 = trajectory[i-1]
                            lat2, lon2 = trajectory[i]
                            
                            # Simple bearing calculation
                            dlon = math.radians(lon2 - lon1)
                            lat1_rad = math.radians(lat1)
                            lat2_rad = math.radians(lat2)
                            
                            y = math.sin(dlon) * math.cos(lat2_rad)
                            x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
                                 math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
                            bearing = math.degrees(math.atan2(y, x))
                            
                            # Add small arrow marker
                            folium.RegularPolygonMarker(
                                location=trajectory[i],
                                number_of_sides=3,
                                radius=4,
                                rotation=bearing,
                                color=colors[status],
                                fill=True,
                                fillColor=colors[status],
                                fillOpacity=0.7,
                                popup=f"Direction: {bearing:.0f}°"
                            ).add_to(m)
            
            # Add vessel marker at most recent position
            if len(trajectory) > 0:
                last_pos = trajectory[-1]
                
                # Create detailed popup
                popup_html = f"""
                <div style="font-family: Arial; font-size: 12px; width: 200px;">
                    <h4 style="margin: 0; color: {colors[status]};">Vessel {mmsi}</h4>
                    <hr style="margin: 5px 0;">
                    <b>Status:</b> {status_text}<br>
                    <b>Fishing Activity:</b> {result.fishing_summary.fishing_percentage:.1f}%<br>
                    <b>Risk Level:</b> {result.risk_level}<br>
                    <b>Data Points:</b> {item['data_points']}<br>
                    <b>Time Period:</b> {result.fishing_summary.time_period_hours:.1f}h
                </div>
                """
                
                # Choose appropriate icon
                if status == 'violation':
                    icon_color = 'red'
                    icon_name = 'exclamation-triangle'
                elif status == 'fishing':
                    icon_color = 'orange'
                    icon_name = 'ship'
                else:
                    icon_color = 'green'
                    icon_name = 'ship'
                
                folium.Marker(
                    location=last_pos,
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(
                        color=icon_color,
                        icon=icon_name,
                        prefix='fa'
                    )
                ).add_to(m)
                
                vessels_added += 1
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 250px; height: 160px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;">
        <h4 style="margin-top: 0;">Vessel Status & Trajectory</h4>
        <i class="fa fa-ship" style="color: {colors['clean']}"></i> Clean (No violations)<br>
        <i class="fa fa-ship" style="color: {colors['fishing']}"></i> Fishing Activity<br>
        <i class="fa fa-exclamation-triangle" style="color: {colors['violation']}"></i> Illegal Fishing<br>
        <hr style="margin: 5px 0;">
        <i class="fa fa-circle" style="color: lightgreen"></i> Trip Start<br>
        <i class="fa fa-circle" style="color: lightcoral"></i> Trip End<br>
        <i class="fa fa-play" style="color: gray"></i> Direction Arrows<br>
        <small>Lines show vessel trajectories over time</small>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add info about vessels displayed
        if vessels_added == 0:
            st.warning("No vessels could be displayed on the map due to invalid coordinate data.")
        else:
            st.info(f"Displaying {vessels_added} vessel(s) on the map.")
        
        return m
    
    def display_summary_stats(self, results: list):
        """Display summary statistics."""
        
        # Calculate statistics
        total_vessels = len(results)
        successful_analyses = sum(1 for r in results if r['status'] == 'success')
        vessels_with_violations = sum(1 for r in results if r['status'] == 'success' and r['result'] and r['result'].violation_detected)
        
        if successful_analyses > 0:
            avg_fishing_activity = sum(r['result'].fishing_summary.fishing_percentage 
                                     for r in results if r['status'] == 'success' and r['result']) / successful_analyses
        else:
            avg_fishing_activity = 0
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total Vessels",
                value=total_vessels,
                delta=f"{successful_analyses} analyzed"
            )
        
        with col2:
            st.metric(
                label="🎣 Avg Fishing Activity",
                value=f"{avg_fishing_activity:.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                label="🚨 Violations Detected",
                value=vessels_with_violations,
                delta=f"{vessels_with_violations/total_vessels*100:.1f}%" if total_vessels > 0 else "0%"
            )
        
        with col4:
            st.metric(
                label="✅ Success Rate",
                value=f"{successful_analyses/total_vessels*100:.0f}%" if total_vessels > 0 else "0%",
                delta=None
            )
    
    def display_vessel_details(self, results: list):
        """Display detailed results for each vessel."""
        
        st.subheader("📋 Vessel Analysis Details")
        
        for item in results:
            mmsi = item['mmsi']
            status = item['status']
            
            with st.expander(f"🚢 Vessel {mmsi} - {status.upper()}", expanded=False):
                
                if status == 'error':
                    st.error(f"❌ Analysis failed: {item['message']}")
                    continue
                
                result = item['result']
                
                # Status indicator
                if result.violation_detected:
                    st.markdown("""
                    <div class="violation-alert">
                        🚨 <b>ILLEGAL FISHING DETECTED</b><br>
                        This vessel was found fishing in Marine Protected Areas
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="clean-alert">
                        ✅ <b>NO VIOLATIONS DETECTED</b><br>
                        This vessel appears to be operating legally
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**🎣 Fishing Activity**")
                    st.write(f"Fishing Points: {result.fishing_summary.fishing_points}")
                    st.write(f"Total Points: {result.fishing_summary.total_ais_points}")
                    st.write(f"Fishing %: {result.fishing_summary.fishing_percentage:.1f}%")
                
                with col2:
                    st.write("**⚠️ Risk Assessment**")
                    st.write(f"Risk Level: {result.risk_level}")
                    st.write(f"Risk Score: {result.risk_score:.3f}")
                    st.write(f"Data Points: {item['data_points']}")
                
                with col3:
                    st.write("**📅 Time Period**")
                    st.write(f"Start: {result.time_period_start.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"End: {result.time_period_end.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"Duration: {result.fishing_summary.time_period_hours:.1f}h")
                
                # MPA violations if any
                if result.mpa_analysis and result.mpa_analysis.violations:
                    st.write("**🚨 MPA Violations:**")
                    for violation in result.mpa_analysis.violations:
                        st.write(f"- {violation.mpa_name}: {violation.fishing_points} fishing points ({violation.severity} severity)")

    def run(self):
        """Run the Streamlit dashboard."""
        
        # Header
        st.markdown('<h1 class="main-header">🌊 Ocean Monitoring Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("**Illegal Fishing Detection System** - Monitor vessel activities and detect violations in Marine Protected Areas")
        
        # Sidebar
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # Initialize detector
            if not self.initialize_detector():
                st.stop()
            
            # Input section
            st.subheader("📝 Vessel Input")
            mmsi_input = st.text_area(
                "Enter MMSI numbers (comma-separated):",
                placeholder="12639560807591, 51394439323066, 77182424306278",
                help="Enter vessel MMSI numbers separated by commas"
            )
            
            # Analysis settings
            st.subheader("⚙️ Settings")
            max_points = st.slider("Max data points per vessel", 100, 2000, 500, 100)
            
            # Analyze button
            analyze_clicked = st.button("🔍 Analyze Vessels", type="primary")
            
            # Model info
            with st.expander("🤖 Model Information"):
                try:
                    model_info = self.detector.get_model_info()
                    st.write(f"**Model Version:** {model_info.get('model_version', 'Unknown')}")
                    st.write(f"**Training Accuracy:** {model_info.get('training_accuracy', 'Unknown')}")
                    st.write(f"**Features:** {model_info.get('feature_count', 'Unknown')}")
                except Exception as e:
                    st.write(f"**Error loading model info:** {str(e)}")
        
        # Main content
        if analyze_clicked and mmsi_input.strip():
            
            # Parse MMSI list
            mmsi_list = self.parse_mmsi_list(mmsi_input)
            
            if not mmsi_list:
                st.error("Please enter valid MMSI numbers")
                st.stop()
            
            st.success(f"Analyzing {len(mmsi_list)} vessel(s)...")
            
            # Analyze vessels
            results = self.analyze_vessels(mmsi_list)
            
            # Display results
            if results:
                
                # Summary statistics
                self.display_summary_stats(results)
                
                st.markdown("---")
                
                # Interactive map
                st.subheader("🗺️ Vessel Tracking Map")
                
                try:
                    map_obj = self.create_combined_map(results)
                    
                    # Display the map with error handling
                    map_data = st_folium(
                        map_obj, 
                        width=None, 
                        height=500,
                        returned_objects=["last_object_clicked"],
                        debug=False
                    )
                    
                    # Display clicked vessel info if any
                    if map_data and 'last_object_clicked' in map_data and map_data['last_object_clicked']:
                        st.info(f"Last clicked: {map_data['last_object_clicked']}")
                        
                except Exception as e:
                    st.error(f"Error creating map: {str(e)}")
                    st.write("**Debug info:**")
                    st.write(f"Number of results: {len(results)}")
                    st.write(f"Successful analyses: {sum(1 for r in results if r['status'] == 'success')}")
                    
                    # Try to create a simple fallback map
                    try:
                        st.write("Attempting to create a simple fallback map...")
                        import folium
                        fallback_map = folium.Map(location=[0, 0], zoom_start=2)
                        folium.Marker([0, 0], popup="Fallback Map").add_to(fallback_map)
                        st_folium(fallback_map, height=300)
                        st.info("Fallback map displayed. Please check your data and try again.")
                    except Exception as fallback_error:
                        st.error(f"Even fallback map failed: {str(fallback_error)}")
                        st.write("Please check your streamlit_folium installation.")
                
                st.markdown("---")
                
                # Detailed results
                self.display_vessel_details(results)
                
            else:
                st.error("No vessels could be analyzed")
        
        elif analyze_clicked:
            st.warning("Please enter MMSI numbers to analyze")
        
        else:
            # Welcome message
            st.info("""
Enter MMSI numbers to analyze.
            """)


def main():
    """Main function to run the Streamlit app."""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()