"""
Evaluator for RL system performance
"""

import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch

from .metrics import MetricsCollector
from madison.environment import MadisonEnvironment
from madison.orchestration.coordinator import AgentCoordinator
from rl.value_based.dqn import DQNAgent
from rl.policy_gradient.ppo import PPOAgent


class Evaluator:
    """Evaluates RL system performance"""
    
    def __init__(
        self,
        env: MadisonEnvironment,
        coordinator: AgentCoordinator,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize evaluator
        
        Args:
            env: Environment instance
            coordinator: Agent coordinator
            metrics_collector: Metrics collector instance
        """
        self.env = env
        self.coordinator = coordinator
        self.metrics_collector = metrics_collector or MetricsCollector()
        
    def evaluate_episode(self, training: bool = False) -> Dict[str, Any]:
        """
        Evaluate one episode
        
        Args:
            training: Whether agents are in training mode
            
        Returns:
            Episode evaluation results
        """
        observation = self.env.reset()
        self.coordinator.reset()
        
        episode_rewards = {}
        episode_length = 0
        done = False
        
        while not done:
            actions = self.coordinator.coordinate_step(observation)
            next_observation, rewards, done, info = self.env.step(actions)
            
            for agent_id, reward in rewards.items():
                if agent_id not in episode_rewards:
                    episode_rewards[agent_id] = 0.0
                episode_rewards[agent_id] += reward
            
            observation = next_observation
            episode_length += 1
        
        total_reward = sum(episode_rewards.values())
        
        # Collect metrics
        metrics = {
            'total_reward': total_reward,
            'episode_length': episode_length,
            'agent_rewards': episode_rewards,
            'num_insights': len(self.env.generated_insights),
            'num_analyses': len(self.env.completed_analyses),
            'num_data': len(self.env.collected_data),
            'avg_insight_quality': np.mean([i.get('quality', 0.0) for i in self.env.generated_insights]) if self.env.generated_insights else 0.0
        }
        
        return metrics
    
    def evaluate_multiple_episodes(self, num_episodes: int = 10, training: bool = False) -> Dict[str, Any]:
        """
        Evaluate over multiple episodes
        
        Args:
            num_episodes: Number of episodes to evaluate
            training: Whether agents are in training mode
            
        Returns:
            Aggregated evaluation results
        """
        all_metrics = []
        
        for _ in range(num_episodes):
            episode_metrics = self.evaluate_episode(training=training)
            all_metrics.append(episode_metrics)
            self.metrics_collector.record_episode(episode_metrics)
        
        # Aggregate metrics
        aggregated = {
            'num_episodes': num_episodes,
            'mean_total_reward': np.mean([m['total_reward'] for m in all_metrics]),
            'std_total_reward': np.std([m['total_reward'] for m in all_metrics]),
            'mean_episode_length': np.mean([m['episode_length'] for m in all_metrics]),
            'mean_num_insights': np.mean([m['num_insights'] for m in all_metrics]),
            'mean_num_analyses': np.mean([m['num_analyses'] for m in all_metrics]),
            'mean_num_data': np.mean([m['num_data'] for m in all_metrics]),
            'mean_insight_quality': np.mean([m['avg_insight_quality'] for m in all_metrics]),
            'episode_metrics': all_metrics
        }
        
        return aggregated
    
    def compare_models(
        self,
        model_paths: Dict[str, str],
        num_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Compare different model checkpoints
        
        Args:
            model_paths: Dictionary mapping model names to file paths
            num_episodes: Number of episodes per model
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for model_name, model_path in model_paths.items():
            # Load model (simplified - would need proper model loading)
            print(f"Evaluating {model_name}...")
            
            results = self.evaluate_multiple_episodes(num_episodes, training=False)
            comparison_results[model_name] = results
        
        return comparison_results
    
    def analyze_learning_dynamics(self) -> Dict[str, Any]:
        """Analyze learning dynamics from collected metrics"""
        return self.metrics_collector.get_performance_summary()
    
    def generate_report(self, output_path: str):
        """Generate evaluation report"""
        summary = self.metrics_collector.get_performance_summary()
        
        report = f"""
# Evaluation Report

## Performance Summary

"""
        for metric_name, data in summary.items():
            stats = data.get('statistics', {})
            convergence = data.get('convergence', {})
            
            report += f"### {metric_name}\n"
            report += f"- Mean: {stats.get('mean', 0):.4f}\n"
            report += f"- Std: {stats.get('std', 0):.4f}\n"
            report += f"- Min: {stats.get('min', 0):.4f}\n"
            report += f"- Max: {stats.get('max', 0):.4f}\n"
            
            if convergence.get('converged'):
                report += f"- Converged: Yes\n"
                report += f"- Final Value: {convergence.get('final_value', 0):.4f}\n"
                report += f"- Improvement: {convergence.get('improvement_percent', 0):.2f}%\n"
            else:
                report += f"- Converged: No\n"
            
            report += "\n"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")

