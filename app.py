import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Import custom modules
from data_generator import SmartMeterDataGenerator
from ml_models import EnergyDemandPredictor
from anomaly_detection import AnomalyDetector
from pattern_analysis import PatternAnalyzer
from dashboard_components import DashboardComponents
from utils import Utils

# Page configuration
st.set_page_config(
    page_title="Smart Grid Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = SmartMeterDataGenerator()
if 'predictor' not in st.session_state:
    st.session_state.predictor = EnergyDemandPredictor()
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()
if 'pattern_analyzer' not in st.session_state:
    st.session_state.pattern_analyzer = PatternAnalyzer()
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = DashboardComponents()
if 'utils' not in st.session_state:
    st.session_state.utils = Utils()

def main():
    st.title("⚡ Smart Grid Energy Analytics Platform")
    st.markdown("### Comprehensive Energy Management & Analytics Dashboard")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Dashboard",
        [
            "Real-time Monitoring",
            "Demand Prediction",
            "Anomaly Detection", 
            "Pattern Analysis",
            "Grid Health",
            "Load Balancing",
            "Data Management"
        ]
    )
    
    # Display selected page
    if page == "Real-time Monitoring":
        real_time_monitoring()
    elif page == "Demand Prediction":
        demand_prediction()
    elif page == "Anomaly Detection":
        anomaly_detection()
    elif page == "Pattern Analysis":
        pattern_analysis()
    elif page == "Grid Health":
        grid_health()
    elif page == "Load Balancing":
        load_balancing()
    elif page == "Data Management":
        data_management()

def real_time_monitoring():
    st.header("📊 Real-time Energy Monitoring")
    
    # Generate real-time data
    current_data = st.session_state.data_generator.generate_realtime_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Consumption",
            f"{current_data['total_consumption']:.2f} MW",
            f"{current_data['consumption_change']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Peak Demand",
            f"{current_data['peak_demand']:.2f} MW",
            f"{current_data['peak_change']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Grid Efficiency",
            f"{current_data['efficiency']:.1f}%",
            f"{current_data['efficiency_change']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Active Meters",
            f"{current_data['active_meters']:,}",
            f"{current_data['meter_change']:+d}"
        )
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Flow by Region")
        flow_data = st.session_state.data_generator.generate_flow_data()
        fig = st.session_state.dashboard.create_energy_flow_chart(flow_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Consumption by Sector")
        sector_data = st.session_state.data_generator.generate_sector_data()
        fig = st.session_state.dashboard.create_sector_chart(sector_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series data
    st.subheader("24-Hour Energy Consumption Trend")
    timeseries_data = st.session_state.data_generator.generate_timeseries_data()
    fig = st.session_state.dashboard.create_timeseries_chart(timeseries_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh toggle
    if st.checkbox("Auto-refresh (30 seconds)"):
        time.sleep(30)
        st.rerun()

def demand_prediction():
    st.header("🔮 Energy Demand Prediction")
    
    # Model parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Model Configuration")
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
        model_type = st.selectbox("Model Type", ["ARIMA", "Prophet", "Linear Regression"])
        include_weather = st.checkbox("Include Weather Data", True)
        include_seasonal = st.checkbox("Include Seasonal Patterns", True)
    
    with col2:
        # Generate historical data for training
        historical_data = st.session_state.data_generator.generate_historical_data(days=90)
        
        # Train and predict
        if st.button("Generate Prediction"):
            with st.spinner("Training model and generating predictions..."):
                predictions = st.session_state.predictor.predict_demand(
                    historical_data, 
                    forecast_days, 
                    model_type,
                    include_weather,
                    include_seasonal
                )
                
                # Display prediction chart
                fig = st.session_state.dashboard.create_prediction_chart(historical_data, predictions)
                st.plotly_chart(fig, use_container_width=True)
                
                # Model performance metrics
                st.subheader("Model Performance")
                metrics = st.session_state.predictor.get_model_metrics()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col3:
                    st.metric("MAPE", f"{metrics['mape']:.1f}%")
                with col4:
                    st.metric("R²", f"{metrics['r2']:.3f}")
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    feature_data = st.session_state.predictor.get_feature_importance()
    if feature_data is not None:
        fig = st.session_state.dashboard.create_feature_importance_chart(feature_data)
        st.plotly_chart(fig, use_container_width=True)

def anomaly_detection():
    st.header("🚨 Anomaly Detection")
    
    # Detection parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Detection Settings")
        method = st.selectbox("Detection Method", ["Isolation Forest", "Z-Score", "IQR", "Statistical"])
        sensitivity = st.slider("Sensitivity", 0.1, 1.0, 0.5)
        window_size = st.slider("Window Size (hours)", 1, 24, 6)
        
        if st.button("Run Anomaly Detection"):
            # Generate data with anomalies
            data_with_anomalies = st.session_state.data_generator.generate_data_with_anomalies()
            
            # Detect anomalies
            anomalies = st.session_state.anomaly_detector.detect_anomalies(
                data_with_anomalies, method, sensitivity, window_size
            )
            
            st.session_state.current_anomalies = anomalies
    
    with col2:
        if 'current_anomalies' in st.session_state:
            anomalies = st.session_state.current_anomalies
            
            # Anomaly visualization
            fig = st.session_state.dashboard.create_anomaly_chart(anomalies)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly summary
            st.subheader("Anomaly Summary")
            anomaly_summary = st.session_state.anomaly_detector.get_anomaly_summary(anomalies)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", anomaly_summary['total'])
            with col2:
                st.metric("Power Outages", anomaly_summary['outages'])
            with col3:
                st.metric("Suspected Theft", anomaly_summary['theft'])
            
            # Detailed anomaly table
            st.subheader("Anomaly Details")
            anomaly_df = st.session_state.anomaly_detector.get_anomaly_details(anomalies)
            st.dataframe(anomaly_df)

def pattern_analysis():
    st.header("📈 Consumption Pattern Analysis")
    
    # Generate pattern data
    pattern_data = st.session_state.data_generator.generate_pattern_data()
    
    # Pattern analysis
    patterns = st.session_state.pattern_analyzer.analyze_patterns(pattern_data)
    
    # Consumption patterns by time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Consumption Patterns")
        hourly_fig = st.session_state.dashboard.create_hourly_pattern_chart(patterns['hourly'])
        st.plotly_chart(hourly_fig, use_container_width=True)
    
    with col2:
        st.subheader("Weekly Consumption Patterns")
        weekly_fig = st.session_state.dashboard.create_weekly_pattern_chart(patterns['weekly'])
        st.plotly_chart(weekly_fig, use_container_width=True)
    
    # Seasonal patterns
    st.subheader("Seasonal Analysis")
    seasonal_fig = st.session_state.dashboard.create_seasonal_chart(patterns['seasonal'])
    st.plotly_chart(seasonal_fig, use_container_width=True)
    
    # Customer segmentation
    st.subheader("Customer Segmentation Analysis")
    segmentation = st.session_state.pattern_analyzer.customer_segmentation(pattern_data)
    
    col1, col2 = st.columns(2)
    with col1:
        seg_fig = st.session_state.dashboard.create_segmentation_chart(segmentation)
        st.plotly_chart(seg_fig, use_container_width=True)
    
    with col2:
        st.write("**Segment Characteristics:**")
        for segment, chars in segmentation['characteristics'].items():
            st.write(f"**{segment}:** {chars}")

def grid_health():
    st.header("🏥 Grid Health Monitoring")
    
    # Grid health metrics
    health_data = st.session_state.data_generator.generate_health_data()
    
    # Overall health score
    st.subheader("Grid Health Score")
    health_score = health_data['overall_score']
    
    if health_score >= 90:
        st.success(f"Grid Health: Excellent ({health_score:.1f}/100)")
    elif health_score >= 70:
        st.warning(f"Grid Health: Good ({health_score:.1f}/100)")
    else:
        st.error(f"Grid Health: Needs Attention ({health_score:.1f}/100)")
    
    # Health indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Voltage Stability", f"{health_data['voltage_stability']:.1f}%")
    with col2:
        st.metric("Frequency Control", f"{health_data['frequency_control']:.1f}%")
    with col3:
        st.metric("Load Balance", f"{health_data['load_balance']:.1f}%")
    with col4:
        st.metric("Equipment Health", f"{health_data['equipment_health']:.1f}%")
    
    # Health monitoring charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Voltage Monitoring")
        voltage_fig = st.session_state.dashboard.create_voltage_chart(health_data['voltage_data'])
        st.plotly_chart(voltage_fig, use_container_width=True)
    
    with col2:
        st.subheader("Frequency Monitoring")
        freq_fig = st.session_state.dashboard.create_frequency_chart(health_data['frequency_data'])
        st.plotly_chart(freq_fig, use_container_width=True)
    
    # Equipment status
    st.subheader("Equipment Status")
    equipment_df = pd.DataFrame(health_data['equipment_status'])
    
    # Color-code equipment status
    def color_status(val):
        if val == 'Online':
            return 'background-color: #d4edda'
        elif val == 'Warning':
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_df = equipment_df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df)

def load_balancing():
    st.header("⚖️ Load Balancing Optimization")
    
    # Current load distribution
    load_data = st.session_state.data_generator.generate_load_data()
    
    st.subheader("Current Load Distribution")
    load_fig = st.session_state.dashboard.create_load_distribution_chart(load_data)
    st.plotly_chart(load_fig, use_container_width=True)
    
    # Load balancing recommendations
    recommendations = st.session_state.utils.generate_load_recommendations(load_data)
    
    st.subheader("Optimization Recommendations")
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"Recommendation {i+1}: {rec['title']}"):
            st.write(f"**Impact:** {rec['impact']}")
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Expected Savings:** {rec['savings']}")
            
            if st.button(f"Implement Recommendation {i+1}"):
                st.success("Recommendation implemented successfully!")
                st.info("Load balancing adjustments have been applied to the grid.")
    
    # Load forecasting for optimization
    st.subheader("Load Forecasting for Optimization")
    forecast_data = st.session_state.data_generator.generate_load_forecast()
    forecast_fig = st.session_state.dashboard.create_load_forecast_chart(forecast_data)
    st.plotly_chart(forecast_fig, use_container_width=True)

def data_management():
    st.header("🗄️ Data Management & Configuration")
    
    # Data generation controls
    st.subheader("Smart Meter Data Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_meters = st.number_input("Number of Smart Meters", 100, 10000, 1000)
        data_frequency = st.selectbox("Data Frequency", ["1min", "5min", "15min", "1hour"])
        include_anomalies = st.checkbox("Include Anomalies", True)
    
    with col2:
        st.write("**Data Sources:**")
        st.write("- Smart meter readings")
        st.write("- Weather data integration")
        st.write("- Grid infrastructure data")
        st.write("- Historical consumption patterns")
    
    if st.button("Generate New Dataset"):
        with st.spinner("Generating smart meter data..."):
            new_data = st.session_state.data_generator.generate_comprehensive_dataset(
                num_meters, data_frequency, include_anomalies
            )
            st.success(f"Generated data for {num_meters} meters with {len(new_data)} records")
    
    # Data quality metrics
    st.subheader("Data Quality Metrics")
    quality_metrics = st.session_state.utils.get_data_quality_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Completeness", f"{quality_metrics['completeness']:.1f}%")
    with col2:
        st.metric("Accuracy", f"{quality_metrics['accuracy']:.1f}%")
    with col3:
        st.metric("Timeliness", f"{quality_metrics['timeliness']:.1f}%")
    with col4:
        st.metric("Consistency", f"{quality_metrics['consistency']:.1f}%")
    
    # System configuration
    st.subheader("System Configuration")
    
    with st.expander("Model Parameters"):
        st.write("Configure machine learning model parameters")
        prediction_threshold = st.slider("Prediction Confidence Threshold", 0.5, 0.99, 0.85)
        anomaly_sensitivity = st.slider("Anomaly Detection Sensitivity", 0.1, 1.0, 0.3)
        pattern_window = st.slider("Pattern Analysis Window (days)", 7, 90, 30)
    
    with st.expander("Alert Settings"):
        st.write("Configure system alerts and notifications")
        enable_alerts = st.checkbox("Enable Real-time Alerts", True)
        alert_threshold = st.slider("Alert Threshold (% deviation)", 5, 50, 20)
        notification_email = st.text_input("Notification Email", "admin@smartgrid.com")

if __name__ == "__main__":
    main()
