import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self):
        self.detectors = {}
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, data, method='Isolation Forest', sensitivity=0.5, window_size=6):
        """
        Detect anomalies in energy consumption data using various methods
        """
        consumption = data['consumption'].values
        timestamps = data['timestamp'].values
        
        if method == 'Isolation Forest':
            anomalies = self._isolation_forest_detection(data, sensitivity)
        elif method == 'Z-Score':
            anomalies = self._zscore_detection(data, sensitivity)
        elif method == 'IQR':
            anomalies = self._iqr_detection(data, sensitivity)
        elif method == 'Statistical':
            anomalies = self._statistical_detection(data, sensitivity, window_size)
        else:
            anomalies = self._isolation_forest_detection(data, sensitivity)
        
        # Classify anomaly types
        anomalies = self._classify_anomaly_types(anomalies)
        
        return anomalies
    
    def _isolation_forest_detection(self, data, sensitivity):
        """Detect anomalies using Isolation Forest"""
        # Prepare features
        features = self._prepare_anomaly_features(data)
        
        # Configure contamination based on sensitivity
        contamination = 0.01 + (sensitivity * 0.09)  # 1% to 10% contamination
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features)
        
        # Create result dataframe
        result = data.copy()
        result['is_anomaly'] = anomaly_labels == -1
        result['anomaly_score'] = iso_forest.decision_function(features)
        result['method'] = 'Isolation Forest'
        
        return result
    
    def _zscore_detection(self, data, sensitivity):
        """Detect anomalies using Z-Score method"""
        result = data.copy()
        
        # Calculate z-scores
        consumption = data['consumption'].values
        z_scores = np.abs(stats.zscore(consumption))
        
        # Threshold based on sensitivity (higher sensitivity = lower threshold)
        threshold = 3.5 - (sensitivity * 1.5)  # Range: 2.0 to 3.5
        
        result['is_anomaly'] = z_scores > threshold
        result['anomaly_score'] = z_scores
        result['method'] = 'Z-Score'
        
        return result
    
    def _iqr_detection(self, data, sensitivity):
        """Detect anomalies using Interquartile Range method"""
        result = data.copy()
        
        consumption = data['consumption'].values
        Q1 = np.percentile(consumption, 25)
        Q3 = np.percentile(consumption, 75)
        IQR = Q3 - Q1
        
        # Adjust multiplier based on sensitivity
        multiplier = 2.5 - (sensitivity * 1.0)  # Range: 1.5 to 2.5
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        result['is_anomaly'] = (consumption < lower_bound) | (consumption > upper_bound)
        result['anomaly_score'] = np.maximum(
            (lower_bound - consumption) / IQR,
            (consumption - upper_bound) / IQR
        )
        result['anomaly_score'] = np.maximum(result['anomaly_score'], 0)
        result['method'] = 'IQR'
        
        return result
    
    def _statistical_detection(self, data, sensitivity, window_size):
        """Detect anomalies using statistical methods with rolling windows"""
        result = data.copy()
        consumption = data['consumption'].values
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(consumption).rolling(window=window_size, min_periods=1).mean()
        rolling_std = pd.Series(consumption).rolling(window=window_size, min_periods=1).std()
        
        # Threshold based on sensitivity
        std_multiplier = 3.0 - (sensitivity * 1.0)  # Range: 2.0 to 3.0
        
        upper_bound = rolling_mean + std_multiplier * rolling_std
        lower_bound = rolling_mean - std_multiplier * rolling_std
        
        result['is_anomaly'] = (consumption < lower_bound) | (consumption > upper_bound)
        result['anomaly_score'] = np.maximum(
            (lower_bound - consumption) / rolling_std,
            (consumption - upper_bound) / rolling_std
        ).fillna(0)
        result['anomaly_score'] = np.maximum(result['anomaly_score'], 0)
        result['method'] = 'Statistical'
        
        return result
    
    def _classify_anomaly_types(self, anomalies):
        """Classify detected anomalies into different types"""
        result = anomalies.copy()
        
        # Initialize anomaly type
        result['anomaly_type'] = 'normal'
        
        anomaly_mask = result['is_anomaly']
        if not anomaly_mask.any():
            return result
        
        anomalous_consumption = result.loc[anomaly_mask, 'consumption']
        mean_consumption = result['consumption'].mean()
        
        # Classify based on consumption patterns
        for idx in result[anomaly_mask].index:
            consumption = result.loc[idx, 'consumption']
            hour = result.loc[idx, 'hour']
            
            if consumption < mean_consumption * 0.3:
                # Very low consumption could indicate outage
                result.loc[idx, 'anomaly_type'] = 'outage'
            elif consumption < mean_consumption * 0.7:
                # Moderately low consumption could indicate theft
                result.loc[idx, 'anomaly_type'] = 'theft'
            elif consumption > mean_consumption * 1.5:
                # High consumption could indicate surge or equipment malfunction
                if hour in [18, 19, 20]:  # Peak hours
                    result.loc[idx, 'anomaly_type'] = 'peak_surge'
                else:
                    result.loc[idx, 'anomaly_type'] = 'equipment_malfunction'
            else:
                result.loc[idx, 'anomaly_type'] = 'unknown'
        
        return result
    
    def _prepare_anomaly_features(self, data):
        """Prepare features for anomaly detection"""
        features = []
        
        # Basic consumption
        consumption = data['consumption'].values
        features.append(consumption)
        
        # Time-based features
        features.append(data['hour'].values)
        features.append(data['day_of_week'].values)
        
        # Rolling statistics
        rolling_mean_24h = pd.Series(consumption).rolling(window=24, min_periods=1).mean().values
        rolling_std_24h = pd.Series(consumption).rolling(window=24, min_periods=1).std().fillna(0).values
        features.append(rolling_mean_24h)
        features.append(rolling_std_24h)
        
        # Lag features
        for lag in [1, 24]:
            lag_feature = np.concatenate([np.full(lag, consumption[0]), consumption[:-lag]])
            features.append(lag_feature)
        
        # Rate of change
        rate_of_change = np.concatenate([[0], np.diff(consumption)])
        features.append(rate_of_change)
        
        feature_matrix = np.column_stack(features)
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix_scaled
    
    def get_anomaly_summary(self, anomalies):
        """Get summary statistics of detected anomalies"""
        total_anomalies = anomalies['is_anomaly'].sum()
        
        # Count by type
        anomaly_types = anomalies[anomalies['is_anomaly']]['anomaly_type'].value_counts()
        
        outages = anomaly_types.get('outage', 0)
        theft = anomaly_types.get('theft', 0)
        surges = anomaly_types.get('peak_surge', 0) + anomaly_types.get('equipment_malfunction', 0)
        
        summary = {
            'total': int(total_anomalies),
            'outages': int(outages),
            'theft': int(theft),
            'surges': int(surges),
            'percentage': (total_anomalies / len(anomalies)) * 100 if len(anomalies) > 0 else 0
        }
        
        return summary
    
    def get_anomaly_details(self, anomalies):
        """Get detailed information about detected anomalies"""
        anomaly_details = anomalies[anomalies['is_anomaly']].copy()
        
        if len(anomaly_details) == 0:
            return pd.DataFrame(columns=['Timestamp', 'Consumption', 'Anomaly Type', 'Severity', 'Method'])
        
        # Create severity categories
        anomaly_details['severity'] = pd.cut(
            anomaly_details['anomaly_score'],
            bins=[0, 1, 2, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        
        # Format for display
        result = pd.DataFrame({
            'Timestamp': anomaly_details['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
            'Consumption (kWh)': anomaly_details['consumption'].round(2),
            'Anomaly Type': anomaly_details['anomaly_type'].str.title(),
            'Severity': anomaly_details['severity'],
            'Anomaly Score': anomaly_details['anomaly_score'].round(3),
            'Detection Method': anomaly_details['method']
        })
        
        return result.sort_values('Anomaly Score', ascending=False)
    
    def detect_seasonal_anomalies(self, data, period=24):
        """Detect seasonal anomalies based on periodic patterns"""
        result = data.copy()
        consumption = data['consumption'].values
        
        # Extract seasonal pattern
        seasonal_means = np.zeros(period)
        seasonal_stds = np.zeros(period)
        
        for i in range(period):
            indices = list(range(i, len(consumption), period))
            period_data = [consumption[idx] for idx in indices if idx < len(consumption)]
            
            if period_data:
                seasonal_means[i] = np.mean(period_data)
                seasonal_stds[i] = np.std(period_data)
        
        # Detect anomalies based on seasonal patterns
        anomaly_scores = np.zeros(len(consumption))
        
        for i, value in enumerate(consumption):
            seasonal_idx = i % period
            expected_mean = seasonal_means[seasonal_idx]
            expected_std = seasonal_stds[seasonal_idx]
            
            if expected_std > 0:
                anomaly_scores[i] = abs(value - expected_mean) / expected_std
            else:
                anomaly_scores[i] = 0
        
        # Threshold for seasonal anomalies
        threshold = np.percentile(anomaly_scores, 95)  # Top 5% as anomalies
        
        result['is_seasonal_anomaly'] = anomaly_scores > threshold
        result['seasonal_anomaly_score'] = anomaly_scores
        
        return result
