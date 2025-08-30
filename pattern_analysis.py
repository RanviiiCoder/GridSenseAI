import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class PatternAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clusters = None
        
    def analyze_patterns(self, data):
        """
        Comprehensive pattern analysis of energy consumption data
        """
        patterns = {
            'hourly': self._analyze_hourly_patterns(data),
            'weekly': self._analyze_weekly_patterns(data),
            'monthly': self._analyze_monthly_patterns(data),
            'seasonal': self._analyze_seasonal_patterns(data),
            'correlations': self._analyze_correlations(data)
        }
        
        return patterns
    
    def _analyze_hourly_patterns(self, data):
        """Analyze consumption patterns by hour of day"""
        hourly_stats = data.groupby('hour')['consumption'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Calculate peak and off-peak hours
        peak_threshold = hourly_stats['mean'].quantile(0.8)
        hourly_stats['category'] = hourly_stats['mean'].apply(
            lambda x: 'Peak' if x >= peak_threshold else 
                     'Medium' if x >= hourly_stats['mean'].median() else 'Off-Peak'
        )
        
        # Calculate variability
        hourly_stats['variability'] = hourly_stats['std'] / hourly_stats['mean']
        
        return hourly_stats
    
    def _analyze_weekly_patterns(self, data):
        """Analyze consumption patterns by day of week"""
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        weekly_stats = data.groupby('day_of_week')['consumption'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        weekly_stats['weekday_name'] = weekly_stats['day_of_week'].apply(lambda x: weekday_names[x])
        weekly_stats['is_weekend'] = weekly_stats['day_of_week'] >= 5
        
        # Calculate weekend vs weekday differences
        weekend_avg = weekly_stats[weekly_stats['is_weekend']]['mean'].mean()
        weekday_avg = weekly_stats[~weekly_stats['is_weekend']]['mean'].mean()
        weekly_stats['weekend_difference'] = weekend_avg - weekday_avg
        
        return weekly_stats
    
    def _analyze_monthly_patterns(self, data):
        """Analyze consumption patterns by month"""
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_stats = data.groupby('month')['consumption'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        monthly_stats['month_name'] = monthly_stats['month'].apply(lambda x: month_names[x-1])
        
        # Identify seasonal categories
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        monthly_stats['season'] = monthly_stats['month'].apply(get_season)
        
        return monthly_stats
    
    def _analyze_seasonal_patterns(self, data):
        """Analyze seasonal consumption patterns"""
        # Add season column
        data_with_season = data.copy()
        data_with_season['season'] = data_with_season['month'].apply(
            lambda x: 'Winter' if x in [12, 1, 2] else
                     'Spring' if x in [3, 4, 5] else
                     'Summer' if x in [6, 7, 8] else 'Fall'
        )
        
        seasonal_stats = data_with_season.groupby('season')['consumption'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Calculate seasonal indices (relative to annual average)
        annual_avg = data['consumption'].mean()
        seasonal_stats['seasonal_index'] = seasonal_stats['mean'] / annual_avg
        
        # Add temperature correlation if available
        if 'temperature' in data.columns:
            temp_correlation = {}
            for season in seasonal_stats['season']:
                season_data = data_with_season[data_with_season['season'] == season]
                if len(season_data) > 1:
                    corr, _ = pearsonr(season_data['consumption'], season_data['temperature'])
                    temp_correlation[season] = corr
                else:
                    temp_correlation[season] = 0
            
            seasonal_stats['temperature_correlation'] = seasonal_stats['season'].map(temp_correlation)
        
        return seasonal_stats
    
    def _analyze_correlations(self, data):
        """Analyze correlations between different variables"""
        correlations = {}
        
        # Time-based correlations
        numeric_columns = ['consumption', 'hour', 'day_of_week', 'month']
        
        if 'temperature' in data.columns:
            numeric_columns.append('temperature')
        
        correlation_matrix = data[numeric_columns].corr()
        correlations['correlation_matrix'] = correlation_matrix
        
        # Lag correlations (consumption with its own lags)
        lag_correlations = {}
        consumption = data['consumption'].values
        
        for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
            if len(consumption) > lag:
                lagged_values = consumption[:-lag]
                current_values = consumption[lag:]
                corr, _ = pearsonr(current_values, lagged_values)
                lag_correlations[f'lag_{lag}_hours'] = corr
        
        correlations['lag_correlations'] = lag_correlations
        
        return correlations
    
    def customer_segmentation(self, data):
        """Perform customer segmentation based on consumption patterns"""
        if 'customer_type' not in data.columns:
            return {'segments': {}, 'characteristics': {}}
        
        # Aggregate consumption data by customer type
        customer_features = data.groupby('customer_type').agg({
            'consumption': ['mean', 'std', 'max', 'min'],
            'hour': lambda x: x.mode().iloc[0] if len(x) > 0 else 0,  # Most common hour
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['customer_type', 'avg_consumption', 'std_consumption', 
                                   'max_consumption', 'min_consumption', 'peak_hour']
        
        # Calculate additional metrics
        customer_features['variability'] = customer_features['std_consumption'] / customer_features['avg_consumption']
        customer_features['consumption_range'] = customer_features['max_consumption'] - customer_features['min_consumption']
        
        # Create characteristics description
        characteristics = {}
        for _, row in customer_features.iterrows():
            customer_type = row['customer_type']
            
            if customer_type == 'Low':
                desc = f"Consistent low consumption (avg: {row['avg_consumption']:.1f} kWh), peak at {row['peak_hour']}h"
            elif customer_type == 'Medium':
                desc = f"Moderate consumption (avg: {row['avg_consumption']:.1f} kWh), variability: {row['variability']:.2f}"
            else:  # High
                desc = f"High consumption (avg: {row['avg_consumption']:.1f} kWh), wide range: {row['consumption_range']:.1f} kWh"
            
            characteristics[customer_type] = desc
        
        # Calculate segment sizes
        segment_sizes = data['customer_type'].value_counts().to_dict()
        
        segmentation = {
            'segments': segment_sizes,
            'characteristics': characteristics,
            'detailed_stats': customer_features
        }
        
        return segmentation
    
    def detect_consumption_trends(self, data):
        """Detect long-term consumption trends"""
        # Convert timestamp to numeric for trend analysis
        data_sorted = data.sort_values('timestamp')
        data_sorted['timestamp_numeric'] = pd.to_numeric(data_sorted['timestamp'])
        
        # Calculate rolling averages for trend detection
        data_sorted['rolling_7d'] = data_sorted['consumption'].rolling(window=168, min_periods=1).mean()  # 7 days
        data_sorted['rolling_30d'] = data_sorted['consumption'].rolling(window=720, min_periods=1).mean()  # 30 days
        
        # Calculate trend slopes
        trends = {}
        
        # Overall trend
        if len(data_sorted) > 1:
            overall_trend = np.polyfit(range(len(data_sorted)), data_sorted['consumption'], 1)[0]
            trends['overall_trend'] = {
                'slope': overall_trend,
                'direction': 'increasing' if overall_trend > 0 else 'decreasing' if overall_trend < 0 else 'stable',
                'magnitude': abs(overall_trend)
            }
        
        # Monthly trends
        monthly_trends = {}
        for month in range(1, 13):
            month_data = data_sorted[data_sorted['month'] == month]
            if len(month_data) > 1:
                trend_slope = np.polyfit(range(len(month_data)), month_data['consumption'], 1)[0]
                monthly_trends[month] = trend_slope
        
        trends['monthly_trends'] = monthly_trends
        
        return trends
    
    def identify_peak_events(self, data, threshold_percentile=95):
        """Identify peak consumption events"""
        threshold = np.percentile(data['consumption'], threshold_percentile)
        
        peak_events = data[data['consumption'] >= threshold].copy()
        peak_events = peak_events.sort_values('consumption', ascending=False)
        
        # Analyze peak event patterns
        peak_analysis = {
            'total_events': len(peak_events),
            'threshold_value': threshold,
            'average_peak_consumption': peak_events['consumption'].mean(),
            'peak_hours': peak_events['hour'].value_counts().to_dict(),
            'peak_weekdays': peak_events['day_of_week'].value_counts().to_dict(),
            'peak_months': peak_events['month'].value_counts().to_dict() if 'month' in peak_events.columns else {}
        }
        
        # Identify consecutive peak events (potential grid stress periods)
        peak_events_sorted = peak_events.sort_values('timestamp')
        consecutive_events = []
        
        if len(peak_events_sorted) > 1:
            current_streak = 1
            max_streak = 1
            
            for i in range(1, len(peak_events_sorted)):
                time_diff = (peak_events_sorted.iloc[i]['timestamp'] - 
                           peak_events_sorted.iloc[i-1]['timestamp']).total_seconds() / 3600
                
                if time_diff <= 2:  # Within 2 hours
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1
        
        peak_analysis['max_consecutive_events'] = max_streak if len(peak_events_sorted) > 1 else 0
        peak_analysis['events_data'] = peak_events.head(20)  # Top 20 peak events
        
        return peak_analysis
    
    def analyze_load_factor(self, data):
        """Calculate and analyze load factor patterns"""
        # Load factor = Average Load / Peak Load
        
        # Daily load factors
        daily_data = data.groupby(data['timestamp'].dt.date).agg({
            'consumption': ['mean', 'max']
        }).reset_index()
        
        daily_data.columns = ['date', 'avg_consumption', 'peak_consumption']
        daily_data['load_factor'] = daily_data['avg_consumption'] / daily_data['peak_consumption']
        
        # Overall load factor statistics
        load_factor_stats = {
            'average_load_factor': daily_data['load_factor'].mean(),
            'load_factor_std': daily_data['load_factor'].std(),
            'min_load_factor': daily_data['load_factor'].min(),
            'max_load_factor': daily_data['load_factor'].max(),
            'load_factor_trend': np.polyfit(range(len(daily_data)), daily_data['load_factor'], 1)[0]
        }
        
        # Categorize load factor performance
        if load_factor_stats['average_load_factor'] > 0.7:
            load_factor_stats['performance'] = 'Excellent'
        elif load_factor_stats['average_load_factor'] > 0.5:
            load_factor_stats['performance'] = 'Good'
        elif load_factor_stats['average_load_factor'] > 0.3:
            load_factor_stats['performance'] = 'Fair'
        else:
            load_factor_stats['performance'] = 'Poor'
        
        return load_factor_stats
