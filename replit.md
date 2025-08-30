# Smart Grid Energy Analytics Platform

## Overview

This is a comprehensive Smart Grid Energy Analytics Platform built with Streamlit that provides real-time monitoring, predictive analytics, and pattern analysis for energy consumption data. The platform simulates smart meter data and offers multiple machine learning-based analysis capabilities including demand forecasting, anomaly detection, and consumption pattern analysis. It features an interactive dashboard with visualizations for energy flow, consumption patterns, and operational recommendations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit-based Web Application**: Single-page application with multiple dashboard views
- **Interactive Visualizations**: Plotly-based charts and graphs for real-time data display
- **Multi-page Navigation**: Sidebar-based navigation between different analytics modules
- **Responsive Layout**: Wide layout configuration optimized for dashboard viewing

### Backend Architecture
- **Modular Python Architecture**: Separate modules for different functionalities (data generation, ML models, anomaly detection, pattern analysis)
- **Object-Oriented Design**: Each major component implemented as a class with specific responsibilities
- **Session State Management**: Streamlit session state used to maintain component instances across page interactions

### Data Processing Pipeline
- **Data Generation Layer**: `SmartMeterDataGenerator` class simulates realistic energy consumption data with regional and sectoral variations
- **Feature Engineering**: Automated creation of time-based features (hour, day of week, month) and weather simulation
- **Data Preprocessing**: StandardScaler integration for ML model input preparation

### Machine Learning Architecture
- **Multi-Model Support**: Support for Random Forest, Linear Regression, and placeholder for ARIMA/Prophet models
- **Anomaly Detection Engine**: Multiple detection methods including Isolation Forest, Z-Score, IQR, and Statistical approaches
- **Pattern Analysis**: Time-series pattern recognition with clustering and correlation analysis
- **Model Evaluation**: Comprehensive metrics calculation (MAE, RMSE, R²) for prediction accuracy

### Visualization Framework
- **Plotly Integration**: Custom dashboard components using Plotly Express and Graph Objects
- **Real-time Updates**: Dynamic chart updates with configurable refresh intervals
- **Multi-chart Layouts**: Subplot arrangements for comprehensive data visualization
- **Color-coded Themes**: Consistent color scheme across all visualizations

### Analytics Modules
- **Real-time Monitoring**: Live dashboard with key performance indicators and energy flow visualization
- **Demand Prediction**: ML-based forecasting with configurable time horizons and model selection
- **Anomaly Detection**: Multi-method anomaly identification with sensitivity controls
- **Pattern Analysis**: Temporal pattern discovery (hourly, daily, weekly, seasonal)
- **Recommendations Engine**: Automated operational recommendations based on data analysis

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for dashboard interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Scikit-learn**: Machine learning algorithms and preprocessing tools

### Visualization
- **Plotly Express & Graph Objects**: Interactive plotting and visualization
- **Plotly Subplots**: Multi-panel chart layouts

### Statistical Analysis
- **SciPy**: Statistical functions and correlation analysis

### Machine Learning Models
- **RandomForestRegressor**: Primary prediction model for demand forecasting
- **LinearRegression**: Alternative prediction model
- **IsolationForest**: Anomaly detection algorithm
- **KMeans**: Clustering for pattern analysis
- **StandardScaler**: Feature normalization

### Data Handling
- **DateTime**: Time series data processing and manipulation
- **Random**: Data simulation and variability generation

### Development Tools
- **Warnings**: Error suppression for cleaner output

Note: The application currently uses simulated data generation rather than connecting to external databases or APIs. Future enhancements could include integration with real smart meter APIs or time-series databases.
