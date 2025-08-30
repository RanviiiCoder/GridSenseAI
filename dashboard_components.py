import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class DashboardComponents:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def create_energy_flow_chart(self, flow_data):
        """Create energy flow visualization by region"""
        fig = go.Figure()
        
        # Add consumption bars
        fig.add_trace(go.Bar(
            x=flow_data['region'],
            y=flow_data['consumption'],
            name='Consumption',
            marker_color=self.colors['primary'],
            opacity=0.8
        ))
        
        # Add generation bars
        fig.add_trace(go.Bar(
            x=flow_data['region'],
            y=flow_data['generation'],
            name='Generation',
            marker_color=self.colors['success'],
            opacity=0.8
        ))
        
        # Add net flow line
        fig.add_trace(go.Scatter(
            x=flow_data['region'],
            y=flow_data['net_flow'],
            mode='lines+markers',
            name='Net Flow',
            line=dict(color=self.colors['warning'], width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Energy Flow by Region',
            xaxis_title='Region',
            yaxis_title='Energy (MW)',
            yaxis2=dict(
                title='Net Flow (MW)',
                overlaying='y',
                side='right'
            ),
            barmode='group',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_sector_chart(self, sector_data):
        """Create consumption by sector pie chart"""
        fig = go.Figure(data=[go.Pie(
            labels=sector_data['sector'],
            values=sector_data['consumption'],
            hole=0.3,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                line=dict(color='#FFFFFF', width=2)
            )
        )])
        
        fig.update_layout(
            title='Energy Consumption by Sector',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_timeseries_chart(self, timeseries_data):
        """Create time series consumption chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timeseries_data['timestamp'],
            y=timeseries_data['consumption'],
            mode='lines',
            name='Consumption',
            line=dict(color=self.colors['primary'], width=2),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        # Add moving average
        moving_avg = timeseries_data['consumption'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(
            x=timeseries_data['timestamp'],
            y=moving_avg,
            mode='lines',
            name='3-Hour Moving Average',
            line=dict(color=self.colors['warning'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='24-Hour Energy Consumption Trend',
            xaxis_title='Time',
            yaxis_title='Consumption (MW)',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_prediction_chart(self, historical_data, predictions):
        """Create demand prediction visualization"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['timestamp'],
            y=historical_data['consumption'],
            mode='lines',
            name='Historical Data',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=predictions['timestamp'],
            y=predictions['predicted_consumption'],
            mode='lines',
            name='Prediction',
            line=dict(color=self.colors['danger'], width=2)
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=predictions['timestamp'].tolist() + predictions['timestamp'].tolist()[::-1],
            y=predictions['upper_bound'].tolist() + predictions['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(214, 39, 40, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Energy Demand Prediction',
            xaxis_title='Time',
            yaxis_title='Consumption (MW)',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_data):
        """Create feature importance bar chart"""
        fig = go.Figure(data=[
            go.Bar(
                x=feature_data['importance'],
                y=feature_data['feature'],
                orientation='h',
                marker_color=self.colors['success'],
                text=feature_data['importance'].round(3),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_anomaly_chart(self, anomalies):
        """Create anomaly detection visualization"""
        fig = go.Figure()
        
        # Normal data points
        normal_data = anomalies[~anomalies['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['consumption'],
            mode='lines',
            name='Normal Consumption',
            line=dict(color=self.colors['primary'], width=1),
            opacity=0.7
        ))
        
        # Anomalous data points
        anomaly_data = anomalies[anomalies['is_anomaly']]
        
        # Color code by anomaly type
        colors = {
            'outage': '#d62728',
            'theft': '#ff7f0e', 
            'peak_surge': '#2ca02c',
            'equipment_malfunction': '#9467bd',
            'unknown': '#8c564b'
        }
        
        for anomaly_type in anomaly_data['anomaly_type'].unique():
            type_data = anomaly_data[anomaly_data['anomaly_type'] == anomaly_type]
            fig.add_trace(go.Scatter(
                x=type_data['timestamp'],
                y=type_data['consumption'],
                mode='markers',
                name=f'Anomaly: {anomaly_type.title()}',
                marker=dict(
                    color=colors.get(anomaly_type, '#8c564b'),
                    size=10,
                    symbol='diamond'
                )
            ))
        
        fig.update_layout(
            title='Anomaly Detection Results',
            xaxis_title='Time',
            yaxis_title='Consumption (MW)',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def create_hourly_pattern_chart(self, hourly_data):
        """Create hourly consumption pattern chart"""
        fig = go.Figure()
        
        # Mean consumption
        fig.add_trace(go.Scatter(
            x=hourly_data['hour'],
            y=hourly_data['mean'],
            mode='lines+markers',
            name='Average Consumption',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=8)
        ))
        
        # Add error bars for standard deviation
        fig.add_trace(go.Scatter(
            x=hourly_data['hour'].tolist() + hourly_data['hour'].tolist()[::-1],
            y=(hourly_data['mean'] + hourly_data['std']).tolist() + 
              (hourly_data['mean'] - hourly_data['std']).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev',
            showlegend=True
        ))
        
        # Color code peak/off-peak hours
        peak_hours = hourly_data[hourly_data['category'] == 'Peak']['hour']
        fig.add_trace(go.Scatter(
            x=peak_hours,
            y=hourly_data[hourly_data['category'] == 'Peak']['mean'],
            mode='markers',
            name='Peak Hours',
            marker=dict(color=self.colors['danger'], size=12, symbol='star')
        ))
        
        fig.update_layout(
            title='Hourly Consumption Patterns',
            xaxis_title='Hour of Day',
            yaxis_title='Average Consumption (MW)',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_weekly_pattern_chart(self, weekly_data):
        """Create weekly consumption pattern chart"""
        fig = go.Figure(data=[
            go.Bar(
                x=weekly_data['weekday_name'],
                y=weekly_data['mean'],
                marker_color=[self.colors['warning'] if weekend else self.colors['primary'] 
                             for weekend in weekly_data['is_weekend']],
                text=weekly_data['mean'].round(1),
                textposition='auto',
                name='Average Consumption'
            )
        ])
        
        fig.update_layout(
            title='Weekly Consumption Patterns',
            xaxis_title='Day of Week',
            yaxis_title='Average Consumption (MW)',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_seasonal_chart(self, seasonal_data):
        """Create seasonal analysis chart"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Seasonal Consumption', 'Seasonal Index'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Seasonal consumption
        fig.add_trace(
            go.Bar(x=seasonal_data['season'], y=seasonal_data['mean'],
                  marker_color=self.colors['primary'], name='Average Consumption'),
            row=1, col=1
        )
        
        # Seasonal index
        fig.add_trace(
            go.Scatter(x=seasonal_data['season'], y=seasonal_data['seasonal_index'],
                      mode='lines+markers', marker_color=self.colors['warning'],
                      name='Seasonal Index'),
            row=1, col=2
        )
        
        # Add horizontal line at 1.0 for seasonal index
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
        
        fig.update_layout(
            title='Seasonal Analysis',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_segmentation_chart(self, segmentation):
        """Create customer segmentation chart"""
        segments = segmentation['segments']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(segments.keys()),
                values=list(segments.values()),
                hole=0.3,
                textinfo='label+value+percent',
                textposition='outside',
                marker=dict(
                    colors=[self.colors['success'], self.colors['primary'], self.colors['warning']],
                    line=dict(color='#FFFFFF', width=2)
                )
            )
        ])
        
        fig.update_layout(
            title='Customer Segmentation',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_voltage_chart(self, voltage_data):
        """Create voltage monitoring chart"""
        fig = go.Figure()
        
        # Voltage line
        fig.add_trace(go.Scatter(
            x=voltage_data['timestamp'],
            y=voltage_data['voltage'],
            mode='lines+markers',
            name='Voltage',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(
                color=[self.colors['danger'] if status == 'Warning' else self.colors['success'] 
                      for status in voltage_data['status']],
                size=6
            )
        ))
        
        # Add safe operating range
        fig.add_hline(y=240, line_dash="dash", line_color="red", 
                     annotation_text="Upper Limit (240V)")
        fig.add_hline(y=220, line_dash="dash", line_color="red", 
                     annotation_text="Lower Limit (220V)")
        
        fig.update_layout(
            title='Voltage Monitoring',
            xaxis_title='Time',
            yaxis_title='Voltage (V)',
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def create_frequency_chart(self, frequency_data):
        """Create frequency monitoring chart"""
        fig = go.Figure()
        
        # Frequency line
        fig.add_trace(go.Scatter(
            x=frequency_data['timestamp'],
            y=frequency_data['frequency'],
            mode='lines+markers',
            name='Frequency',
            line=dict(color=self.colors['success'], width=2),
            marker=dict(
                color=[self.colors['danger'] if status == 'Warning' else self.colors['success'] 
                      for status in frequency_data['status']],
                size=6
            )
        ))
        
        # Add safe operating range
        fig.add_hline(y=50.2, line_dash="dash", line_color="red", 
                     annotation_text="Upper Limit (50.2Hz)")
        fig.add_hline(y=49.8, line_dash="dash", line_color="red", 
                     annotation_text="Lower Limit (49.8Hz)")
        
        fig.update_layout(
            title='Frequency Monitoring',
            xaxis_title='Time',
            yaxis_title='Frequency (Hz)',
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def create_load_distribution_chart(self, load_data):
        """Create load distribution chart"""
        fig = go.Figure()
        
        # Current load bars
        fig.add_trace(go.Bar(
            x=load_data['region'],
            y=load_data['current_load'],
            name='Current Load',
            marker_color=self.colors['primary']
        ))
        
        # Capacity line
        fig.add_trace(go.Scatter(
            x=load_data['region'],
            y=load_data['capacity'],
            mode='lines+markers',
            name='Capacity',
            line=dict(color=self.colors['danger'], width=3),
            marker=dict(size=10)
        ))
        
        # Utilization percentage as text annotations
        for i, row in load_data.iterrows():
            fig.add_annotation(
                x=row['region'],
                y=row['current_load'] + 5,
                text=f"{row['utilization']:.1f}%",
                showarrow=False,
                font=dict(size=12, color='black')
            )
        
        fig.update_layout(
            title='Current Load Distribution by Region',
            xaxis_title='Region',
            yaxis_title='Load (MW)',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_load_forecast_chart(self, forecast_data):
        """Create load forecasting chart"""
        fig = go.Figure()
        
        # Predicted load
        fig.add_trace(go.Scatter(
            x=forecast_data['timestamp'],
            y=forecast_data['predicted_load'],
            mode='lines',
            name='Predicted Load',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Confidence bands
        fig.add_trace(go.Scatter(
            x=forecast_data['timestamp'].tolist() + forecast_data['timestamp'].tolist()[::-1],
            y=forecast_data['upper_bound'].tolist() + forecast_data['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        fig.update_layout(
            title='48-Hour Load Forecast',
            xaxis_title='Time',
            yaxis_title='Predicted Load (MW)',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
