"""
Visualization utilities for RL training and evaluation
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns

sns.set_style("whitegrid")


class Visualizer:
    """Creates visualizations for RL training and evaluation"""
    
    @staticmethod
    def plot_learning_curves(
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None,
        title: str = "Learning Curves"
    ):
        """
        Plot learning curves for multiple metrics
        
        Args:
            metrics: Dictionary of metric names to value lists
            save_path: Path to save figure
            title: Plot title
        """
        num_metrics = len(metrics)
        if num_metrics == 0:
            return
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4))
        if num_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            
            # Plot raw values
            ax.plot(values, alpha=0.3, label='Raw', color='lightblue')
            
            # Plot smoothed curve
            if len(values) > 10:
                window_size = min(50, len(values) // 10)
                smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                x_smooth = np.arange(window_size-1, len(values))
                ax.plot(x_smooth, smoothed, label='Smoothed', color='darkblue', linewidth=2)
            
            ax.set_title(metric_name)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_reward_components(
        reward_components: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Plot breakdown of reward components"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = np.arange(len(list(reward_components.values())[0]))
        
        for component_name, values in reward_components.items():
            ax.plot(episodes, values, label=component_name, alpha=0.7)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Components Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_convergence_analysis(
        metrics: Dict[str, List[float]],
        window_size: int = 50,
        save_path: Optional[str] = None
    ):
        """Plot convergence analysis for metrics"""
        num_metrics = len(metrics)
        if num_metrics == 0:
            return
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4))
        if num_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            
            if len(values) >= window_size:
                # Calculate rolling statistics
                rolling_means = []
                rolling_stds = []
                
                for i in range(window_size, len(values)):
                    window = values[i - window_size:i]
                    rolling_means.append(np.mean(window))
                    rolling_stds.append(np.std(window))
                
                x = np.arange(window_size, len(values))
                ax.plot(x, rolling_means, label='Rolling Mean', linewidth=2)
                ax.fill_between(
                    x,
                    np.array(rolling_means) - np.array(rolling_stds),
                    np.array(rolling_means) + np.array(rolling_stds),
                    alpha=0.3,
                    label='±1 Std'
                )
            
            ax.set_title(f'{metric_name} - Convergence Analysis')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_agent_performance(
        agent_performance: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Plot performance comparison across agents"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agent_names = list(agent_performance.keys())
        num_episodes = len(agent_performance[agent_names[0]])
        episodes = np.arange(num_episodes)
        
        for agent_name, values in agent_performance.items():
            # Smooth the curve
            if len(values) > 10:
                window_size = min(20, len(values) // 5)
                smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                x_smooth = np.arange(window_size-1, len(values))
                ax.plot(x_smooth, smoothed, label=agent_name, linewidth=2, alpha=0.8)
            else:
                ax.plot(episodes, values, label=agent_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Agent Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_evaluation_comparison(
        comparison_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """Plot comparison of different models/configurations"""
        model_names = list(comparison_results.keys())
        metrics = set()
        for results in comparison_results.values():
            metrics.update(results.keys())
        
        metrics = sorted([m for m in metrics if isinstance(comparison_results[model_names[0]].get(m), (int, float))])
        
        num_metrics = len(metrics)
        if num_metrics == 0:
            return
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
        if num_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [comparison_results[model].get(metric, 0) for model in model_names]
            
            bars = ax.bar(model_names, values, alpha=0.7)
            ax.set_title(metric)
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
        
        plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

