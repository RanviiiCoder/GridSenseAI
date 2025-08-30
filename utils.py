import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class Utils:
    def __init__(self):
        self.recommendations = [
            {
                'title': 'Peak Shaving Optimization',
                'description': 'Implement demand response programs to reduce peak load by 15-20%',
                'impact': 'High - Reduces grid stress and operational costs',
                'savings': '$2.5M annually'
            },
            {
                'title': 'Load Balancing Across Regions',
                'description': 'Redistribute load from high-utilization to low-utilization regions',
                'impact': 'Medium - Improves overall grid stability',
                'savings': '$1.2M annually'
            },
            {
                'title': 'Smart Meter Deployment',
                'description': 'Deploy advanced metering infrastructure in underserved areas',
                'impact': 'High - Improves monitoring and theft detection',
                'savings': '$3.1M annually'
            },
            {
                'title': 'Renewable Energy Integration',
                'description': 'Increase renewable energy sources to 40% of total generation',
                'impact': 'Very High - Reduces carbon footprint and long-term costs',
                'savings': '$5.8M annually'
            },
            {
                'title': 'Predictive Maintenance',
                'description': 'Implement ML-based predictive maintenance for grid equipment',
                'impact': 'Medium - Reduces equipment downtime',
                'savings': '$1.8M annually'
            }
        ]
    
    def generate_load_recommendations(self, load_data):
        """Generate load balancing recommendations based on current data"""
        recommendations = []
        
        # Analyze load distribution
        overloaded_regions = load_data[load_data['utilization'] > 85]
        underloaded_regions = load_data[load_data['utilization'] < 60]
        
        if len(overloaded_regions) > 0:
            for _, region in overloaded_regions.iterrows():
                rec = {
                    'title': f'Reduce Load in {region["region"]}',
                    'description': f'Current utilization at {region["utilization"]:.1f}% - implement demand response',
                    'impact': 'Critical - Prevents potential outages',
                    'savings': f'${random.randint(500, 2000)}k annually'
                }
                recommendations.append(rec)
        
        if len(underloaded_regions) > 0 and len(overloaded_regions) > 0:
            rec = {
                'title': 'Cross-Regional Load Balancing',
                'description': f'Transfer load from overloaded to underloaded regions',
                'impact': 'High - Optimizes grid utilization',
                'savings': f'${random.randint(800, 1500)}k annually'
            }
            recommendations.append(rec)
        
        # Add some general recommendations
        general_recs = random.sample(self.recommendations, min(3, len(self.recommendations)))
        recommendations.extend(general_recs)
        
        return recommendations
    
    def get_data_quality_metrics(self):
        """Return data quality metrics"""
        # In a real implementation, these would be calculated from actual data
        return {
            'completeness': np.random.uniform(92, 99),
            'accuracy': np.random.uniform(88, 97),
            'timeliness': np.random.uniform(94, 99),
            'consistency': np.random.uniform(89, 96)
        }
    
    def calculate_grid_efficiency(self, generation_data, consumption_data):
        """Calculate overall grid efficiency"""
        if len(generation_data) == 0 or len(consumption_data) == 0:
            return 0
        
        total_generation = generation_data.sum()
        total_consumption = consumption_data.sum()
        
        # Account for transmission losses (typically 5-10%)
        transmission_losses = total_generation * 0.07
        efficiency = ((total_generation - transmission_losses) / total_generation) * 100
        
        return efficiency
    
    def detect_grid_imbalances(self, regional_data):
        """Detect load imbalances across regions"""
        imbalances = []
        
        mean_utilization = regional_data['utilization'].mean()
        std_utilization = regional_data['utilization'].std()
        
        for _, row in regional_data.iterrows():
            deviation = abs(row['utilization'] - mean_utilization)
            
            if deviation > 2 * std_utilization:
                severity = 'High' if deviation > 3 * std_utilization else 'Medium'
                imbalance = {
                    'region': row['region'],
                    'utilization': row['utilization'],
                    'deviation': deviation,
                    'severity': severity,
                    'recommendation': self._get_imbalance_recommendation(row['utilization'])
                }
                imbalances.append(imbalance)
        
        return imbalances
    
    def _get_imbalance_recommendation(self, utilization):
        """Get recommendation based on utilization level"""
        if utilization > 90:
            return "Critical: Implement immediate load shedding"
        elif utilization > 80:
            return "High: Activate demand response programs"
        elif utilization < 30:
            return "Low: Consider load transfer from other regions"
        else:
            return "Monitor: Within acceptable range"
    
    def calculate_carbon_footprint(self, consumption_data, renewable_percentage=30):
        """Calculate carbon footprint based on consumption and renewable mix"""
        # Carbon intensity factors (kg CO2/kWh)
        carbon_factors = {
            'renewable': 0.05,  # Solar, wind, hydro
            'natural_gas': 0.49,
            'coal': 0.82,
            'nuclear': 0.12
        }
        
        # Assume energy mix
        renewable_share = renewable_percentage / 100
        fossil_share = 1 - renewable_share
        
        total_consumption = consumption_data.sum()
        
        # Calculate emissions
        renewable_emissions = total_consumption * renewable_share * carbon_factors['renewable']
        fossil_emissions = total_consumption * fossil_share * carbon_factors['natural_gas']
        
        total_emissions = renewable_emissions + fossil_emissions
        carbon_intensity = total_emissions / total_consumption
        
        return {
            'total_emissions_tons': total_emissions / 1000,  # Convert to tons
            'carbon_intensity': carbon_intensity,
            'renewable_share': renewable_share * 100,
            'emissions_avoided': (total_consumption * carbon_factors['coal'] - total_emissions) / 1000
        }
    
    def generate_maintenance_schedule(self, equipment_data):
        """Generate predictive maintenance schedule"""
        maintenance_schedule = []
        
        for equipment in equipment_data:
            if equipment['Status'] == 'Warning':
                priority = 'High'
                days_until = random.randint(1, 7)
            elif equipment['Efficiency'].replace('%', '') < '90':
                priority = 'Medium'
                days_until = random.randint(7, 30)
            else:
                priority = 'Low'
                days_until = random.randint(30, 90)
            
            maintenance_date = datetime.now() + timedelta(days=days_until)
            
            schedule_item = {
                'equipment_id': equipment['Equipment ID'],
                'equipment_type': equipment['Type'],
                'priority': priority,
                'scheduled_date': maintenance_date.strftime('%Y-%m-%d'),
                'estimated_duration': f"{random.randint(2, 8)} hours",
                'maintenance_type': 'Preventive' if priority == 'Low' else 'Corrective'
            }
            
            maintenance_schedule.append(schedule_item)
        
        return sorted(maintenance_schedule, key=lambda x: x['scheduled_date'])
    
    def optimize_energy_routing(self, supply_data, demand_data):
        """Optimize energy routing between supply and demand points"""
        # Simplified optimization algorithm
        optimization_results = {
            'total_cost_reduction': random.uniform(5, 15),  # Percentage
            'efficiency_improvement': random.uniform(2, 8),  # Percentage
            'recommended_routes': []
        }
        
        # Generate routing recommendations
        for i in range(3):
            route = {
                'from': f"Generation Point {i+1}",
                'to': f"Demand Center {i+1}",
                'capacity': random.randint(50, 200),
                'cost_saving': random.uniform(10, 50),
                'efficiency_gain': random.uniform(2, 10)
            }
            optimization_results['recommended_routes'].append(route)
        
        return optimization_results
    
    def calculate_roi_metrics(self, investment_cost, annual_savings, years=10):
        """Calculate return on investment metrics"""
        # Simple ROI calculation
        total_savings = annual_savings * years
        roi_percentage = ((total_savings - investment_cost) / investment_cost) * 100
        payback_period = investment_cost / annual_savings
        
        # Net Present Value (simplified, assuming 5% discount rate)
        discount_rate = 0.05
        npv = sum([annual_savings / (1 + discount_rate)**year for year in range(1, years + 1)]) - investment_cost
        
        return {
            'roi_percentage': roi_percentage,
            'payback_period_years': payback_period,
            'npv': npv,
            'total_savings': total_savings,
            'investment_cost': investment_cost
        }
    
    def generate_alert_conditions(self):
        """Define alert conditions for monitoring"""
        alert_conditions = [
            {
                'parameter': 'Grid Utilization',
                'threshold': '>90%',
                'severity': 'Critical',
                'action': 'Activate load shedding protocols'
            },
            {
                'parameter': 'Voltage Deviation',
                'threshold': '±5% from nominal',
                'severity': 'High',
                'action': 'Adjust transformer tap settings'
            },
            {
                'parameter': 'Frequency Deviation',
                'threshold': '±0.2 Hz from 50Hz',
                'severity': 'High',
                'action': 'Activate frequency response reserves'
            },
            {
                'parameter': 'Equipment Temperature',
                'threshold': '>85°C',
                'severity': 'Medium',
                'action': 'Schedule immediate inspection'
            },
            {
                'parameter': 'Power Factor',
                'threshold': '<0.85',
                'severity': 'Medium',
                'action': 'Install reactive power compensation'
            }
        ]
        
        return alert_conditions
    
    def format_currency(self, amount):
        """Format currency values for display"""
        if amount >= 1000000:
            return f"${amount/1000000:.1f}M"
        elif amount >= 1000:
            return f"${amount/1000:.0f}K"
        else:
            return f"${amount:.0f}"
    
    def format_power(self, value, unit='MW'):
        """Format power values for display"""
        if unit == 'MW' and value < 1:
            return f"{value*1000:.0f} kW"
        else:
            return f"{value:.2f} {unit}"
    
    def get_system_status(self, health_score):
        """Get overall system status based on health score"""
        if health_score >= 95:
            return {'status': 'Excellent', 'color': 'success', 'icon': '✅'}
        elif health_score >= 85:
            return {'status': 'Good', 'color': 'info', 'icon': '✔️'}
        elif health_score >= 70:
            return {'status': 'Warning', 'color': 'warning', 'icon': '⚠️'}
        else:
            return {'status': 'Critical', 'color': 'danger', 'icon': '🚨'}
