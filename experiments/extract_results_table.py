#!/usr/bin/env python3
"""
Extract comprehensive results table from all training configurations
"""

import json
import re
from pathlib import Path
import numpy as np
import yaml
from collections import defaultdict

def extract_rewards_from_stdout(stdout: str) -> dict:
    """Extract reward statistics from training stdout"""
    if not stdout:
        return {}
    
    lines = stdout.split('\n')
    reward_lines = [l for l in lines if 'Average Reward' in l]
    
    if not reward_lines:
        return {}
    
    rewards = []
    for line in reward_lines:
        match = re.search(r'Average Reward\s*=\s*([\d.]+)', line)
        if match:
            rewards.append(float(match.group(1)))
    
    if not rewards:
        return {}
    
    rewards = np.array(rewards)
    
    # Calculate early, middle, late thirds
    n = len(rewards)
    if n >= 3:
        third = n // 3
        early = rewards[:third]
        middle = rewards[third:2*third]
        late = rewards[2*third:]
    else:
        early = middle = late = rewards
    
    return {
        'all_rewards': rewards.tolist(),
        'first': float(rewards[0]) if len(rewards) > 0 else None,
        'last': float(rewards[-1]) if len(rewards) > 0 else None,
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
        'early_mean': float(np.mean(early)),
        'late_mean': float(np.mean(late)),
        'trend': float(np.mean(late) - np.mean(early)),
        'trend_pct': float((np.mean(late) - np.mean(early)) / np.mean(early) * 100) if np.mean(early) > 0 else 0,
        'num_evaluations': len(rewards)
    }

def get_config_info(config_path: str) -> dict:
    """Get configuration information"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        return {
            'episodes': config['training']['num_episodes'],
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'epsilon_decay': config['training']['epsilon_decay'],
            'buffer_size': config['training']['replay_buffer_size'],
            'target_update': config['training']['target_update_frequency'],
            'network': config['network']['hidden_layers'],
            'ppo_epochs': config.get('ppo', {}).get('ppo_epochs', 4),
            'num_agents': (
                config['agents']['num_intelligence_agents'] +
                config['agents']['num_data_collectors'] +
                config['agents']['num_insight_generators']
            ),
            'num_data_sources': config['environment']['num_data_sources'],
            'num_task_types': config['environment']['num_task_types']
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_results():
    """Analyze all training results and create comprehensive table"""
    
    # Load training results
    results_file = Path('models/training_results.json')
    if not results_file.exists():
        print("❌ training_results.json not found!")
        return
    
    with open(results_file) as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Extract data for each config
    table_data = []
    
    configs = [
        'config/config_simple_2000.yaml',
        'config/config_simple_5000.yaml',
        'config/config_simple_10000.yaml',
        'config/config_medium_2000.yaml',
        'config/config_medium_5000.yaml',
        'config/config_medium_10000.yaml',
        'config/config_complex_2000.yaml',
        'config/config_complex_5000.yaml',
        'config/config_complex_10000.yaml',
    ]
    
    for config_path in configs:
        config_name = Path(config_path).stem.replace('config_', '')
        
        # Find result for this config
        result = None
        for r in results:
            if r.get('config', '').endswith(config_name) or config_name in r.get('config', ''):
                result = r
                break
        
        if not result:
            continue
        
        # Get config info
        config_info = get_config_info(config_path)
        
        # Extract reward stats
        reward_stats = {}
        if result.get('status') == 'success':
            reward_stats = extract_rewards_from_stdout(result.get('stdout', ''))
        
        # Combine data
        row = {
            'config': config_name,
            'status': result.get('status', 'unknown'),
            'time_seconds': result.get('time', 0),
            'episodes': config_info.get('episodes', 0),
            'learning_rate': config_info.get('learning_rate', 0),
            'epsilon_decay': config_info.get('epsilon_decay', 0),
            'buffer_size': config_info.get('buffer_size', 0),
            'target_update': config_info.get('target_update', 0),
            'batch_size': config_info.get('batch_size', 0),
            'ppo_epochs': config_info.get('ppo_epochs', 0),
            'network': str(config_info.get('network', [])),
            'num_agents': config_info.get('num_agents', 0),
            'num_data_sources': config_info.get('num_data_sources', 0),
            'num_task_types': config_info.get('num_task_types', 0),
            **reward_stats
        }
        
        table_data.append(row)
    
    # Sort by complexity and episodes
    def sort_key(row):
        complexity_order = {'simple': 1, 'medium': 2, 'complex': 3}
        parts = row['config'].split('_')
        complexity = parts[0] if len(parts) > 0 else 'unknown'
        episodes = int(parts[1]) if len(parts) > 1 else 0
        return (complexity_order.get(complexity, 99), episodes)
    
    table_data.sort(key=sort_key)
    
    # Print table
    print("\n" + "="*120)
    print("COMPREHENSIVE TRAINING RESULTS TABLE")
    print("="*120)
    print()
    
    # Main results table
    print(f"{'Config':<20} {'Status':<8} {'Episodes':<9} {'Time (min)':<12} {'Mean Reward':<13} {'Min':<8} {'Max':<8} {'Std':<8} {'Trend':<10}")
    print("-"*120)
    
    for row in table_data:
        config = row['config']
        status = "✅" if row['status'] == 'success' else "❌"
        episodes = row['episodes']
        time_min = f"{row['time_seconds']/60:.2f}"
        mean_reward = f"{row.get('mean', 0):.1f}" if row.get('mean') else "N/A"
        min_reward = f"{row.get('min', 0):.1f}" if row.get('min') else "N/A"
        max_reward = f"{row.get('max', 0):.1f}" if row.get('max') else "N/A"
        std_reward = f"{row.get('std', 0):.1f}" if row.get('std') else "N/A"
        trend = row.get('trend', 0)
        trend_str = f"{trend:+.1f}" if trend else "N/A"
        
        print(f"{config:<20} {status:<8} {episodes:<9} {time_min:<12} {mean_reward:<13} {min_reward:<8} {max_reward:<8} {std_reward:<8} {trend_str:<10}")
    
    print("\n" + "="*120)
    print("DETAILED CONVERGENCE ANALYSIS")
    print("="*120)
    print()
    
    print(f"{'Config':<20} {'First Reward':<13} {'Last Reward':<13} {'Early Mean':<13} {'Late Mean':<13} {'Trend %':<10} {'Eval Count':<11}")
    print("-"*120)
    
    for row in table_data:
        config = row['config']
        first = f"{row.get('first', 0):.1f}" if row.get('first') else "N/A"
        last = f"{row.get('last', 0):.1f}" if row.get('last') else "N/A"
        early = f"{row.get('early_mean', 0):.1f}" if row.get('early_mean') else "N/A"
        late = f"{row.get('late_mean', 0):.1f}" if row.get('late_mean') else "N/A"
        trend_pct = f"{row.get('trend_pct', 0):+.1f}%" if row.get('trend_pct') is not None else "N/A"
        eval_count = row.get('num_evaluations', 0)
        
        print(f"{config:<20} {first:<13} {last:<13} {early:<13} {late:<13} {trend_pct:<10} {eval_count:<11}")
    
    print("\n" + "="*120)
    print("HYPERPARAMETER CONFIGURATION")
    print("="*120)
    print()
    
    print(f"{'Config':<20} {'LR':<8} {'Epsilon Decay':<14} {'Buffer':<8} {'Batch':<7} {'Target Up':<11} {'PPO Epochs':<11} {'Network':<25}")
    print("-"*120)
    
    for row in table_data:
        config = row['config']
        lr = f"{row['learning_rate']:.4f}"
        eps_decay = f"{row['epsilon_decay']:.4f}"
        buffer = f"{row['buffer_size']}"
        batch = f"{row['batch_size']}"
        target = f"{row['target_update']}"
        ppo = f"{row['ppo_epochs']}"
        network = row['network'][:24]
        
        print(f"{config:<20} {lr:<8} {eps_decay:<14} {buffer:<8} {batch:<7} {target:<11} {ppo:<11} {network:<25}")
    
    print("\n" + "="*120)
    print("ENVIRONMENT CONFIGURATION")
    print("="*120)
    print()
    
    print(f"{'Config':<20} {'Agents':<8} {'Data Sources':<13} {'Task Types':<11}")
    print("-"*120)
    
    for row in table_data:
        config = row['config']
        agents = f"{row['num_agents']}"
        sources = f"{row['num_data_sources']}"
        tasks = f"{row['num_task_types']}"
        
        print(f"{config:<20} {agents:<8} {sources:<13} {tasks:<11}")
    
    # Save to JSON
    output_file = Path('models/results_table.json')
    with open(output_file, 'w') as f:
        json.dump(table_data, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*120)
    print("SUMMARY STATISTICS BY COMPLEXITY")
    print("="*120)
    print()
    
    for complexity in ['simple', 'medium', 'complex']:
        complex_rows = [r for r in table_data if r['config'].startswith(complexity)]
        if not complex_rows:
            continue
        
        rewards = [r.get('mean', 0) for r in complex_rows if r.get('mean')]
        if rewards:
            print(f"{complexity.upper()} Configs:")
            print(f"  Mean Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
            print(f"  Range: {np.min(rewards):.1f} - {np.max(rewards):.1f}")
            print(f"  Count: {len(complex_rows)}")
            print()

if __name__ == '__main__':
    analyze_results()

