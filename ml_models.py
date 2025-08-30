import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EnergyDemandPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        
    def predict_demand(self, historical_data, forecast_days, model_type='RandomForest', 
                      include_weather=True, include_seasonal=True):
        """
        Predict energy demand for the specified number of days
        """
        # Prepare features
        features = self._prepare_features(historical_data, include_weather, include_seasonal)
        target = historical_data['consumption'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'Linear Regression':
            self.model = LinearRegression()
        else:
            # Default to Random Forest for other types (ARIMA, Prophet would need different implementation)
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        self._calculate_metrics(y_test, y_pred)
        
        # Generate predictions for future days
        predictions = self._generate_future_predictions(historical_data, forecast_days, 
                                                       include_weather, include_seasonal)
        
        return predictions
    
    def _prepare_features(self, data, include_weather, include_seasonal):
        """Prepare feature matrix for ML model"""
        features = []
        feature_names = ['hour', 'day_of_week']
        
        # Basic time features
        features.append(data['hour'].values)
        features.append(data['day_of_week'].values)
        
        # Seasonal features
        if include_seasonal:
            features.append(data['month'].values)
            features.append(data['is_weekend'].astype(int).values)
            feature_names.extend(['month', 'is_weekend'])
        
        # Weather features
        if include_weather:
            features.append(data['temperature'].values)
            feature_names.append('temperature')
        
        # Lag features
        consumption = data['consumption'].values
        for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
            if len(consumption) > lag:
                lag_feature = np.concatenate([np.full(lag, consumption[0]), consumption[:-lag]])
                features.append(lag_feature)
                feature_names.append(f'consumption_lag_{lag}')
        
        # Rolling averages
        for window in [24, 168]:  # 1 day, 1 week
            rolling_avg = pd.Series(consumption).rolling(window=window, min_periods=1).mean().values
            features.append(rolling_avg)
            feature_names.append(f'consumption_rolling_{window}')
        
        self.feature_names = feature_names
        return np.column_stack(features)
    
    def _generate_future_predictions(self, historical_data, forecast_days, 
                                   include_weather, include_seasonal):
        """Generate predictions for future time periods"""
        last_timestamp = historical_data['timestamp'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=forecast_days * 24,
            freq='H'
        )
        
        predictions = []
        last_consumption = historical_data['consumption'].iloc[-1]
        
        for i, timestamp in enumerate(future_timestamps):
            # Create feature vector for this timestamp
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            is_weekend = day_of_week >= 5
            
            # Simulate temperature (in real scenario, use weather API)
            temperature = 20 + 10 * np.sin(2 * np.pi * timestamp.dayofyear / 365) + np.random.normal(0, 2)
            
            # Build feature vector
            features = [hour, day_of_week]
            
            if include_seasonal:
                features.extend([month, int(is_weekend)])
            
            if include_weather:
                features.append(temperature)
            
            # Add lag features (simplified)
            features.extend([last_consumption] * 3)  # lag features
            features.extend([last_consumption] * 2)  # rolling averages
            
            # Make prediction
            feature_vector = np.array(features).reshape(1, -1)
            if len(feature_vector[0]) == len(self.feature_names):
                feature_vector_scaled = self.scaler.transform(feature_vector)
                predicted_consumption = self.model.predict(feature_vector_scaled)[0]
            else:
                # Fallback prediction
                predicted_consumption = last_consumption * self._get_hourly_multiplier(hour)
            
            # Add some uncertainty
            confidence = np.random.uniform(0.7, 0.95)
            uncertainty = predicted_consumption * (1 - confidence) * 0.5
            
            predictions.append({
                'timestamp': timestamp,
                'predicted_consumption': predicted_consumption,
                'confidence': confidence,
                'lower_bound': predicted_consumption - uncertainty,
                'upper_bound': predicted_consumption + uncertainty
            })
            
            # Update last consumption for next iteration
            last_consumption = predicted_consumption * np.random.uniform(0.95, 1.05)
        
        return pd.DataFrame(predictions)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate model performance metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        self.metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def get_model_metrics(self):
        """Return model performance metrics"""
        return self.metrics
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_data = {
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }
            df = pd.DataFrame(importance_data)
            return df.sort_values('importance', ascending=False)
        
        return None
    
    def _get_hourly_multiplier(self, hour):
        """Get consumption multiplier based on hour of day"""
        hourly_pattern = {
            0: 0.6, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.6,
            6: 0.8, 7: 1.0, 8: 1.2, 9: 1.1, 10: 1.0, 11: 1.0,
            12: 1.1, 13: 1.0, 14: 0.9, 15: 0.9, 16: 1.0, 17: 1.2,
            18: 1.4, 19: 1.5, 20: 1.3, 21: 1.1, 22: 0.9, 23: 0.7
        }
        return hourly_pattern.get(hour, 1.0)

class TimeSeriesForecaster:
    """Advanced time series forecasting class"""
    
    def __init__(self):
        self.models = {}
    
    def arima_forecast(self, data, periods=24):
        """ARIMA forecasting (simplified implementation)"""
        # In a real implementation, you would use statsmodels ARIMA
        # This is a simplified version for demonstration
        
        consumption = data['consumption'].values
        
        # Simple trend and seasonality extraction
        trend = np.polyfit(range(len(consumption)), consumption, 1)
        seasonal_pattern = self._extract_seasonal_pattern(consumption, 24)
        
        forecasts = []
        for i in range(periods):
            trend_value = trend[0] * (len(consumption) + i) + trend[1]
            seasonal_value = seasonal_pattern[i % 24]
            forecast = trend_value + seasonal_value + np.random.normal(0, np.std(consumption) * 0.1)
            forecasts.append(forecast)
        
        return np.array(forecasts)
    
    def prophet_forecast(self, data, periods=24):
        """Prophet-like forecasting (simplified implementation)"""
        # Simplified Prophet-like approach
        consumption = data['consumption'].values
        timestamps = data['timestamp'].values
        
        # Extract components
        yearly_seasonality = self._extract_yearly_seasonality(data)
        weekly_seasonality = self._extract_weekly_seasonality(data)
        daily_seasonality = self._extract_daily_seasonality(data)
        
        # Generate forecasts
        forecasts = []
        last_timestamp = timestamps[-1]
        
        for i in range(periods):
            future_timestamp = last_timestamp + pd.Timedelta(hours=i+1)
            
            yearly_component = yearly_seasonality[future_timestamp.dayofyear % 365]
            weekly_component = weekly_seasonality[future_timestamp.weekday()]
            daily_component = daily_seasonality[future_timestamp.hour]
            
            forecast = np.mean(consumption) + yearly_component + weekly_component + daily_component
            forecast += np.random.normal(0, np.std(consumption) * 0.05)
            
            forecasts.append(forecast)
        
        return np.array(forecasts)
    
    def _extract_seasonal_pattern(self, data, period):
        """Extract seasonal pattern from data"""
        pattern = np.zeros(period)
        for i in range(period):
            indices = list(range(i, len(data), period))
            if indices:
                pattern[i] = np.mean([data[idx] for idx in indices if idx < len(data)])
        
        # Remove the mean to get just the seasonal component
        pattern = pattern - np.mean(pattern)
        return pattern
    
    def _extract_yearly_seasonality(self, data):
        """Extract yearly seasonal pattern"""
        yearly_pattern = np.zeros(365)
        
        for _, row in data.iterrows():
            day_of_year = row['timestamp'].dayofyear - 1
            yearly_pattern[day_of_year] = row['consumption']
        
        # Smooth the pattern
        return yearly_pattern - np.mean(yearly_pattern)
    
    def _extract_weekly_seasonality(self, data):
        """Extract weekly seasonal pattern"""
        weekly_pattern = np.zeros(7)
        
        for day in range(7):
            day_data = data[data['day_of_week'] == day]['consumption']
            if len(day_data) > 0:
                weekly_pattern[day] = day_data.mean()
        
        return weekly_pattern - np.mean(weekly_pattern)
    
    def _extract_daily_seasonality(self, data):
        """Extract daily seasonal pattern"""
        daily_pattern = np.zeros(24)
        
        for hour in range(24):
            hour_data = data[data['hour'] == hour]['consumption']
            if len(hour_data) > 0:
                daily_pattern[hour] = hour_data.mean()
        
        return daily_pattern - np.mean(daily_pattern)
