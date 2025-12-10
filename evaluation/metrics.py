"""
Metrics collection and analysis
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


class MetricsCollector:
    """Collects and analyzes performance metrics"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = defaultdict(list)
        self.episode_metrics = []
        
    def record_metric(self, name: str, value: float, episode: int = None):
        """Record a metric value"""
        self.metrics[name].append(value)
        if episode is not None:
            if len(self.episode_metrics) <= episode:
                self.episode_metrics.extend([{}] * (episode + 1 - len(self.episode_metrics)))
            self.episode_metrics[episode][name] = value
    
    def record_episode(self, episode_data: Dict[str, Any]):
        """Record complete episode data"""
        self.episode_metrics.append(episode_data)
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {}
        
        values = self.metrics[metric_name]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values)
        }
    
    def get_convergence_analysis(self, metric_name: str, window_size: int = 50) -> Dict[str, Any]:
        """Analyze convergence of a metric"""
        if metric_name not in self.metrics:
            return {}
        
        values = self.metrics[metric_name]
        
        if len(values) < window_size:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        # Calculate rolling mean
        rolling_means = []
        for i in range(window_size, len(values)):
            window = values[i - window_size:i]
            rolling_means.append(np.mean(window))
        
        # Check for convergence (small variance in recent values)
        if len(rolling_means) > 10:
            recent_std = np.std(rolling_means[-10:])
            recent_mean = np.mean(rolling_means[-10:])
            early_mean = np.mean(rolling_means[:10])
            
            improvement = recent_mean - early_mean
            stability = recent_std / (abs(recent_mean) + 1e-8)
            
            return {
                'converged': stability < 0.1,  # Less than 10% variation
                'stability': stability,
                'improvement': improvement,
                'final_value': recent_mean,
                'initial_value': early_mean,
                'improvement_percent': (improvement / (abs(early_mean) + 1e-8)) * 100
            }
        
        return {'converged': False, 'reason': 'insufficient_data'}
    
    def get_learning_curves(self) -> Dict[str, List[float]]:
        """Get learning curves for all metrics"""
        return dict(self.metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        summary = {}
        
        for metric_name in self.metrics:
            stats = self.get_statistics(metric_name)
            convergence = self.get_convergence_analysis(metric_name)
            summary[metric_name] = {
                'statistics': stats,
                'convergence': convergence
            }
        
        return summary
    
    def compare_periods(self, metric_name: str, period1_end: int, period2_start: int) -> Dict[str, float]:
        """Compare performance between two periods"""
        if metric_name not in self.metrics:
            return {}
        
        values = self.metrics[metric_name]
        
        period1 = values[:period1_end]
        period2 = values[period2_start:]
        
        return {
            'period1_mean': np.mean(period1),
            'period2_mean': np.mean(period2),
            'improvement': np.mean(period2) - np.mean(period1),
            'improvement_percent': ((np.mean(period2) - np.mean(period1)) / (abs(np.mean(period1)) + 1e-8)) * 100
        }

