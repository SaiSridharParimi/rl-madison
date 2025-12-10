"""
Evaluation script for Madison RL system
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from madison.environment import MadisonEnvironment
from madison.orchestration.coordinator import AgentCoordinator
from evaluation.evaluator import Evaluator
from evaluation.visualizer import Visualizer
from evaluation.metrics import MetricsCollector


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Madison RL System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create environment and coordinator
    coordinator = AgentCoordinator(
        num_intelligence_agents=config['agents']['num_intelligence_agents'],
        num_data_collectors=config['agents']['num_data_collectors'],
        num_insight_generators=config['agents']['num_insight_generators']
    )
    
    env = MadisonEnvironment(
        num_data_sources=config['environment']['num_data_sources'],
        num_task_types=config['environment']['num_task_types'],
        coordinator=coordinator
    )
    
    # Create evaluator
    metrics_collector = MetricsCollector()
    evaluator = Evaluator(env, coordinator, metrics_collector)
    
    # Load model if provided
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        # Model loading would go here
    
    # Run evaluation
    print(f"Running evaluation for {args.num_episodes} episodes...")
    results = evaluator.evaluate_multiple_episodes(num_episodes=args.num_episodes, training=False)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Mean Total Reward: {results['mean_total_reward']:.2f} ± {results['std_total_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.2f}")
    print(f"Mean Insights Generated: {results['mean_num_insights']:.2f}")
    print(f"Mean Analyses Completed: {results['mean_num_analyses']:.2f}")
    print(f"Mean Data Collected: {results['mean_num_data']:.2f}")
    print(f"Mean Insight Quality: {results['mean_insight_quality']:.4f}")
    
    # Generate visualizations
    visualizer = Visualizer()
    
    # Learning curves (if we have training data)
    learning_metrics = {
        'total_reward': [r['total_reward'] for r in results['episode_metrics']],
        'num_insights': [r['num_insights'] for r in results['episode_metrics']],
        'insight_quality': [r['avg_insight_quality'] for r in results['episode_metrics']]
    }
    
    visualizer.plot_learning_curves(
        learning_metrics,
        save_path=str(output_dir / 'evaluation_curves.png'),
        title="Evaluation Performance"
    )
    
    # Generate report
    evaluator.generate_report(str(output_dir / 'evaluation_report.md'))
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

