import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SmartMeterDataGenerator:
    def __init__(self):
        self.base_consumption = 1000  # Base consumption in kWh
        self.num_regions = 5
        self.num_sectors = 4
        self.regions = ['North', 'South', 'East', 'West', 'Central']
        self.sectors = ['Residential', 'Commercial', 'Industrial', 'Public']
        
    def generate_realtime_data(self):
        """Generate real-time energy data for dashboard metrics"""
        current_hour = datetime.now().hour
        
        # Simulate daily consumption pattern
        hourly_multiplier = self._get_hourly_multiplier(current_hour)
        
        data = {
            'total_consumption': self.base_consumption * hourly_multiplier * np.random.uniform(0.9, 1.1),
            'consumption_change': np.random.uniform(-5, 8),
            'peak_demand': self.base_consumption * hourly_multiplier * 1.3 * np.random.uniform(0.95, 1.05),
            'peak_change': np.random.uniform(-3, 6),
            'efficiency': np.random.uniform(85, 98),
            'efficiency_change': np.random.uniform(-2, 3),
            'active_meters': np.random.randint(9800, 10200),
            'meter_change': np.random.randint(-20, 30)
        }
        
        return data
    
    def generate_flow_data(self):
        """Generate energy flow data by region"""
        flow_data = []
        for region in self.regions:
            consumption = np.random.uniform(150, 350)
            generation = consumption * np.random.uniform(0.8, 1.2)
            
            flow_data.append({
                'region': region,
                'consumption': consumption,
                'generation': generation,
                'net_flow': generation - consumption,
                'efficiency': min(100, (consumption / generation) * 100) if generation > 0 else 0
            })
        
        return pd.DataFrame(flow_data)
    
    def generate_sector_data(self):
        """Generate consumption data by sector"""
        sector_data = []
        total_consumption = np.random.uniform(800, 1200)
        
        # Sector distribution percentages
        distributions = {
            'Residential': 0.35,
            'Commercial': 0.25,
            'Industrial': 0.30,
            'Public': 0.10
        }
        
        for sector, percentage in distributions.items():
            consumption = total_consumption * percentage * np.random.uniform(0.9, 1.1)
            sector_data.append({
                'sector': sector,
                'consumption': consumption,
                'percentage': (consumption / total_consumption) * 100
            })
        
        return pd.DataFrame(sector_data)
    
    def generate_timeseries_data(self, hours=24):
        """Generate time series data for the last N hours"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        time_range = pd.date_range(start=start_time, end=end_time, freq='H')
        
        timeseries_data = []
        for timestamp in time_range:
            hour = timestamp.hour
            multiplier = self._get_hourly_multiplier(hour)
            
            consumption = self.base_consumption * multiplier * np.random.uniform(0.85, 1.15)
            
            timeseries_data.append({
                'timestamp': timestamp,
                'consumption': consumption,
                'hour': hour,
                'day_of_week': timestamp.weekday()
            })
        
        return pd.DataFrame(timeseries_data)
    
    def generate_historical_data(self, days=90):
        """Generate historical data for ML training"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        historical_data = []
        for timestamp in date_range:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            
            # Base consumption with patterns
            base = self.base_consumption * self._get_hourly_multiplier(hour)
            base *= self._get_weekly_multiplier(day_of_week)
            base *= self._get_seasonal_multiplier(month)
            
            # Add weather effect
            temperature = 20 + 10 * np.sin(2 * np.pi * timestamp.dayofyear / 365) + np.random.normal(0, 5)
            weather_effect = 1 + 0.02 * abs(temperature - 22)  # Heating/cooling effect
            
            consumption = base * weather_effect * np.random.uniform(0.9, 1.1)
            
            historical_data.append({
                'timestamp': timestamp,
                'consumption': consumption,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'temperature': temperature,
                'is_weekend': day_of_week >= 5
            })
        
        return pd.DataFrame(historical_data)
    
    def generate_data_with_anomalies(self):
        """Generate data with anomalies for detection"""
        normal_data = self.generate_timeseries_data(hours=168)  # One week
        
        # Inject anomalies
        anomaly_indices = np.random.choice(len(normal_data), size=int(len(normal_data) * 0.05), replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['outage', 'theft', 'surge'])
            
            if anomaly_type == 'outage':
                normal_data.loc[idx, 'consumption'] *= np.random.uniform(0.1, 0.3)
                normal_data.loc[idx, 'anomaly_type'] = 'outage'
            elif anomaly_type == 'theft':
                normal_data.loc[idx, 'consumption'] *= np.random.uniform(0.6, 0.8)
                normal_data.loc[idx, 'anomaly_type'] = 'theft'
            else:  # surge
                normal_data.loc[idx, 'consumption'] *= np.random.uniform(1.5, 2.0)
                normal_data.loc[idx, 'anomaly_type'] = 'surge'
        
        normal_data['anomaly_type'] = normal_data.get('anomaly_type', 'normal')
        normal_data['is_anomaly'] = normal_data['anomaly_type'] != 'normal'
        
        return normal_data
    
    def generate_pattern_data(self):
        """Generate data for pattern analysis"""
        pattern_data = self.generate_historical_data(days=365)  # Full year
        
        # Add customer categories
        pattern_data['customer_type'] = np.random.choice(
            ['Low', 'Medium', 'High'], 
            size=len(pattern_data),
            p=[0.4, 0.4, 0.2]
        )
        
        # Adjust consumption based on customer type
        type_multipliers = {'Low': 0.5, 'Medium': 1.0, 'High': 2.0}
        for customer_type, multiplier in type_multipliers.items():
            mask = pattern_data['customer_type'] == customer_type
            pattern_data.loc[mask, 'consumption'] *= multiplier * np.random.uniform(0.8, 1.2)
        
        return pattern_data
    
    def generate_health_data(self):
        """Generate grid health monitoring data"""
        health_data = {
            'voltage_stability': np.random.uniform(85, 98),
            'frequency_control': np.random.uniform(88, 99),
            'load_balance': np.random.uniform(82, 96),
            'equipment_health': np.random.uniform(90, 99)
        }
        
        health_data['overall_score'] = np.mean(list(health_data.values()))
        
        # Voltage data over time
        timestamps = pd.date_range(end=datetime.now(), periods=24, freq='H')
        voltage_data = []
        for ts in timestamps:
            voltage = 230 + np.random.normal(0, 5)
            voltage_data.append({
                'timestamp': ts,
                'voltage': voltage,
                'status': 'Normal' if 220 <= voltage <= 240 else 'Warning'
            })
        
        # Frequency data
        frequency_data = []
        for ts in timestamps:
            frequency = 50 + np.random.normal(0, 0.1)
            frequency_data.append({
                'timestamp': ts,
                'frequency': frequency,
                'status': 'Normal' if 49.8 <= frequency <= 50.2 else 'Warning'
            })
        
        # Equipment status
        equipment_types = ['Transformer', 'Circuit Breaker', 'Generator', 'Switch', 'Meter']
        equipment_status = []
        for i, eq_type in enumerate(equipment_types * 4):
            status = np.random.choice(['Online', 'Warning', 'Offline'], p=[0.8, 0.15, 0.05])
            equipment_status.append({
                'Equipment ID': f'{eq_type}-{i//5+1:03d}',
                'Type': eq_type,
                'Status': status,
                'Last Maintenance': (datetime.now() - timedelta(days=np.random.randint(1, 180))).strftime('%Y-%m-%d'),
                'Efficiency': f"{np.random.uniform(85, 99):.1f}%"
            })
        
        health_data['voltage_data'] = pd.DataFrame(voltage_data)
        health_data['frequency_data'] = pd.DataFrame(frequency_data)
        health_data['equipment_status'] = equipment_status
        
        return health_data
    
    def generate_load_data(self):
        """Generate load distribution data"""
        load_data = []
        for region in self.regions:
            current_load = np.random.uniform(80, 120)
            capacity = np.random.uniform(130, 150)
            
            load_data.append({
                'region': region,
                'current_load': current_load,
                'capacity': capacity,
                'utilization': (current_load / capacity) * 100,
                'peak_load': current_load * np.random.uniform(1.2, 1.5)
            })
        
        return pd.DataFrame(load_data)
    
    def generate_load_forecast(self):
        """Generate load forecasting data"""
        hours = 48  # Next 48 hours
        timestamps = pd.date_range(start=datetime.now(), periods=hours, freq='H')
        
        forecast_data = []
        for ts in timestamps:
            hour = ts.hour
            multiplier = self._get_hourly_multiplier(hour)
            
            base_load = 100 * multiplier
            predicted_load = base_load * np.random.uniform(0.9, 1.1)
            confidence = np.random.uniform(0.7, 0.95)
            
            forecast_data.append({
                'timestamp': ts,
                'predicted_load': predicted_load,
                'confidence': confidence,
                'lower_bound': predicted_load * (1 - (1 - confidence) * 0.5),
                'upper_bound': predicted_load * (1 + (1 - confidence) * 0.5)
            })
        
        return pd.DataFrame(forecast_data)
    
    def generate_comprehensive_dataset(self, num_meters, frequency, include_anomalies):
        """Generate comprehensive dataset for analysis"""
        # This would generate a large dataset for the specified number of meters
        # For demonstration, we'll generate a sample
        
        sample_data = []
        for meter_id in range(min(100, num_meters)):  # Limit for demo
            meter_data = self.generate_historical_data(days=30)
            meter_data['meter_id'] = f'METER_{meter_id:04d}'
            sample_data.append(meter_data)
        
        combined_data = pd.concat(sample_data, ignore_index=True)
        
        if include_anomalies:
            # Add some anomalies
            anomaly_count = int(len(combined_data) * 0.02)
            anomaly_indices = np.random.choice(len(combined_data), size=anomaly_count, replace=False)
            combined_data.loc[anomaly_indices, 'consumption'] *= np.random.uniform(0.3, 2.5, size=anomaly_count)
        
        return combined_data
    
    def _get_hourly_multiplier(self, hour):
        """Get consumption multiplier based on hour of day"""
        # Typical daily consumption pattern
        hourly_pattern = {
            0: 0.6, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.6,
            6: 0.8, 7: 1.0, 8: 1.2, 9: 1.1, 10: 1.0, 11: 1.0,
            12: 1.1, 13: 1.0, 14: 0.9, 15: 0.9, 16: 1.0, 17: 1.2,
            18: 1.4, 19: 1.5, 20: 1.3, 21: 1.1, 22: 0.9, 23: 0.7
        }
        return hourly_pattern.get(hour, 1.0)
    
    def _get_weekly_multiplier(self, day_of_week):
        """Get consumption multiplier based on day of week (0=Monday)"""
        # Weekend typically has different patterns
        weekly_pattern = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.8, 6: 0.7}
        return weekly_pattern.get(day_of_week, 1.0)
    
    def _get_seasonal_multiplier(self, month):
        """Get consumption multiplier based on month (seasonal effects)"""
        # Higher consumption in summer (AC) and winter (heating)
        seasonal_pattern = {
            1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9, 6: 1.1,
            7: 1.3, 8: 1.3, 9: 1.0, 10: 0.9, 11: 1.0, 12: 1.2
        }
        return seasonal_pattern.get(month, 1.0)
