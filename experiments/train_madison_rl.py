"""
Training script for Madison RL system
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from madison.environment import MadisonEnvironment
from madison.orchestration.coordinator import AgentCoordinator
from madison.orchestration.task_allocator import TaskAllocator
from rl.value_based.dqn import DQNAgent
from rl.policy_gradient.ppo import PPOAgent
from rl.multi_agent.marl_coordinator import MARLCoordinator
from madison.agents.intelligence_agent import IntelligenceAgent
from madison.agents.data_collector import DataCollector


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_rl_agents(config: dict, state_dim: int, action_dim: int):
    """Create RL agents for different components"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DQN for task allocation
    dqn_allocator = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        epsilon=config['training']['epsilon_start'],
        epsilon_decay=config['training']['epsilon_decay'],
        epsilon_min=config['training']['epsilon_end'],
        batch_size=config['training']['batch_size'],
        replay_buffer_size=config['training']['replay_buffer_size'],
        target_update_frequency=config['training']['target_update_frequency'],
        hidden_dims=config['network']['hidden_layers'],
        device=device
    )
    
    # PPO for agent decision-making
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_epsilon=config['ppo']['clip_epsilon'],
        value_coef=config['ppo']['value_coef'],
        entropy_coef=config['ppo']['entropy_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        ppo_epochs=config['ppo']['ppo_epochs'],
        hidden_dims=config['network']['hidden_layers'],
        device=device
    )
    
    return dqn_allocator, ppo_agent


def train_episode(
    env: MadisonEnvironment,
    coordinator: AgentCoordinator,
    dqn_allocator: DQNAgent,
    ppo_agents: dict,
    marl_coordinator: MARLCoordinator,
    config: dict
) -> dict:
    """Train one episode"""
    observation = env.reset()
    coordinator.reset()
    
    episode_rewards = defaultdict(float)
    episode_losses = defaultdict(list)
    done = False
    
    # Store transitions for RL agents
    transitions = []
    
    while not done:
        # Get actions from coordinator
        actions = coordinator.coordinate_step(observation)
        
        # Execute step
        next_observation, rewards, done, info = env.step(actions)
        
        # Share rewards using MARL coordinator
        team_performance = sum(rewards.values()) / len(rewards) if rewards else 0.0
        shared_rewards = marl_coordinator.share_rewards(rewards, team_performance)
        
        # Update agents with shared rewards
        coordinator.update_agents(shared_rewards, done)
        
        # Update MARL coordinator performance tracking
        for agent_id, reward in shared_rewards.items():
            marl_coordinator.update_performance(agent_id, reward)
        
        # Store transitions for RL training
        # Task allocator (DQN) transitions - only store if we have enough data
        if coordinator.task_allocator.rl_agent and len(dqn_allocator.replay_buffer) < dqn_allocator.batch_size:
            # Skip training until we have enough samples
            pass
        elif coordinator.task_allocator.rl_agent and coordinator.task_allocator.allocation_history:
            # Get the most recent allocation
            last_allocation = coordinator.task_allocator.allocation_history[-1]
            task_id = last_allocation.get('task_id')
            
            # Find the task that was allocated (from previous observation)
            # For now, use current observation's pending tasks or create a representative state
            if observation.get('pending_tasks'):
                task = observation['pending_tasks'][0]
            else:
                # Create a default task structure
                task = {
                    'id': 'default',
                    'type': 'general',
                    'priority': 'medium',
                    'complexity': 0.5,
                    'domain': 'general',
                    'requirements': []
                }
            
            # Get appropriate agents for task type
            task_type = task.get('type', 'general')
            if task_type == 'data_collection':
                available_agents = coordinator.get_agents_by_type('collector')
            elif task_type == 'analysis':
                available_agents = coordinator.get_agents_by_type('intelligence')
            elif task_type == 'synthesis':
                available_agents = coordinator.get_agents_by_type('generator')
            else:
                available_agents = coordinator.agents
            
            # Filter available agents
            available_agents = [a for a in available_agents if coordinator._is_agent_available(a, task)]
            
            if available_agents:
                # Get state using proper state representation
                state = coordinator.task_allocator._get_allocation_state(task, available_agents, observation)
                action = last_allocation.get('action', 0)
                reward = sum(rewards.values()) / len(rewards) if rewards else 0.0
                
                # Get next state
                if next_observation.get('pending_tasks'):
                    next_task = next_observation['pending_tasks'][0]
                    next_task_type = next_task.get('type', 'general')
                    if next_task_type == 'data_collection':
                        next_available_agents = coordinator.get_agents_by_type('collector')
                    elif next_task_type == 'analysis':
                        next_available_agents = coordinator.get_agents_by_type('intelligence')
                    elif next_task_type == 'synthesis':
                        next_available_agents = coordinator.get_agents_by_type('generator')
                    else:
                        next_available_agents = coordinator.agents
                    
                    next_available_agents = [a for a in next_available_agents if coordinator._is_agent_available(a, next_task)]
                    
                    if next_available_agents:
                        next_state = coordinator.task_allocator._get_allocation_state(
                            next_task, next_available_agents, next_observation
                        )
                    else:
                        next_state = np.zeros(15, dtype=np.float32)
                else:
                    next_state = np.zeros(15, dtype=np.float32)
                
                dqn_allocator.store_transition(state, action, reward, next_state, done)
        
        # PPO agent transitions (for intelligence agents)
        for agent in coordinator.get_agents_by_type('intelligence'):
            ppo_agent = ppo_agents.get(agent.agent_id)
            if ppo_agent:
                state = agent.process_observation(observation)
                
                # Use PPO to select action and get log_prob/value
                action, log_prob, value = ppo_agent.select_action(state, training=True)
                
                # Convert action index to action dict
                action_types = ['analyze', 'wait', 'prioritize', 'collaborate', 'explore']
                action_type = action_types[action % len(action_types)]
                action_dict = {'type': action_type, 'target': None, 'priority': 'medium'}
                
                # Store the action for agent
                if agent.agent_id not in actions:
                    actions[agent.agent_id] = action_dict
                
                reward = rewards.get(agent.agent_id, 0.0)
                next_state = agent.process_observation(next_observation)
                
                # Store transition for PPO training
                ppo_agent.store_transition(state, action, reward, log_prob, value, done)
        
        observation = next_observation
        
        # Accumulate rewards
        for agent_id, reward in rewards.items():
            episode_rewards[agent_id] += reward
    
    # Train RL agents
    dqn_loss = dqn_allocator.train_step()
    if dqn_loss:
        episode_losses['dqn'].append(dqn_loss)
    
    for agent_id, ppo_agent in ppo_agents.items():
        ppo_metrics = ppo_agent.train_step()
        if ppo_metrics:
            for key, value in ppo_metrics.items():
                episode_losses[f'ppo_{key}'].append(value)
    
    return {
        'rewards': dict(episode_rewards),
        'total_reward': sum(episode_rewards.values()),
        'losses': {k: np.mean(v) if v else 0.0 for k, v in episode_losses.items()},
        'info': info
    }


def main():
    parser = argparse.ArgumentParser(description='Train Madison RL System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for models (auto-generated from config if not specified)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    num_episodes = args.episodes or config['training']['num_episodes']
    
    # Generate config-specific output directory name
    if args.output_dir is None:
        # Extract config name from path (e.g., 'config_simple_2000.yaml' -> 'simple_2000')
        config_path = Path(args.config)
        config_name = config_path.stem  # Gets filename without extension
        if config_name.startswith('config_'):
            config_name = config_name.replace('config_', '')
        output_dir = Path('models') / config_name
    else:
        output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create RL agents
    state_dim = 15  # State dimension for task allocation
    action_dim = len(coordinator.agents)  # Action = select agent
    
    dqn_allocator, ppo_agent_template = create_rl_agents(config, state_dim, action_dim)
    
    # Set up task allocator with DQN
    task_allocator = TaskAllocator(rl_agent=dqn_allocator)
    coordinator.task_allocator = task_allocator
    
    # Set up PPO agents for intelligence agents
    ppo_agents = {}
    for agent in coordinator.get_agents_by_type('intelligence'):
        state_dim = len(agent.process_observation(env.get_observation()))
        ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=5,  # 5 possible actions: analyze, wait, prioritize, collaborate, explore
            learning_rate=config['training']['learning_rate'],
            gamma=config['training']['gamma'],
            gae_lambda=config['ppo']['gae_lambda'],
            clip_epsilon=config['ppo']['clip_epsilon'],
            value_coef=config['ppo']['value_coef'],
            entropy_coef=config['ppo']['entropy_coef'],
            max_grad_norm=config['ppo']['max_grad_norm'],
            ppo_epochs=config['ppo']['ppo_epochs'],
            hidden_dims=config['network']['hidden_layers']
        )
        # Set up RL policy that uses PPO
        def make_ppo_policy(ppo_agent, base_agent):
            def rl_policy(state, observation):
                action, log_prob, value = ppo_agent.select_action(state, training=True)
                # Store for later use
                base_agent._last_log_prob = log_prob
                base_agent._last_value = value
                # Convert to action dict
                action_types = ['analyze', 'wait', 'prioritize', 'collaborate', 'explore']
                return {'type': action_types[action % len(action_types)], 'target': None, 'priority': 'medium'}
            return rl_policy
        
        agent.rl_policy = make_ppo_policy(ppo_agent, agent)
        ppo_agents[agent.agent_id] = ppo_agent
    
    # Set up Multi-Agent RL coordinator for reward sharing
    marl_coordinator = MARLCoordinator(
        agent_ids=[agent.agent_id for agent in coordinator.agents],
        sharing_strategy='proportional',
        communication_enabled=True
    )
    
    # Training loop
    training_history = {
        'episode_rewards': [],
        'episode_losses': defaultdict(list),
        'episode_metrics': []
    }
    
    print("Starting training...")
    for episode in tqdm(range(num_episodes), desc="Training"):
        episode_results = train_episode(env, coordinator, dqn_allocator, ppo_agents, marl_coordinator, config)
        
        # Record history
        training_history['episode_rewards'].append(episode_results['total_reward'])
        for key, value in episode_results['losses'].items():
            training_history['episode_losses'][key].append(value)
        training_history['episode_metrics'].append(episode_results['info'])
        
        # Evaluation and saving
        if (episode + 1) % config['evaluation']['eval_frequency'] == 0:
            avg_reward = np.mean(training_history['episode_rewards'][-config['evaluation']['eval_frequency']:])
            print(f"\nEpisode {episode + 1}: Average Reward = {avg_reward:.2f}")
        
        if (episode + 1) % config['evaluation']['save_frequency'] == 0:
            dqn_allocator.save(output_dir / f'dqn_allocator_ep{episode+1}.pth')
            for agent_id, ppo_agent in ppo_agents.items():
                ppo_agent.save(output_dir / f'ppo_{agent_id}_ep{episode+1}.pth')
    
    # Save final models
    dqn_allocator.save(output_dir / 'dqn_allocator_final.pth')
    for agent_id, ppo_agent in ppo_agents.items():
        ppo_agent.save(output_dir / f'ppo_{agent_id}_final.pth')
    
    # Generate config info for plot title
    config_path = Path(args.config)
    config_name = config_path.stem.replace('config_', '') if config_path.stem.startswith('config_') else config_path.stem
    num_agents = config['agents']['num_intelligence_agents'] + config['agents']['num_data_collectors'] + config['agents']['num_insight_generators']
    
    # Plot training curves with config-specific filename
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['episode_rewards'], alpha=0.7, linewidth=1)
    # Add smoothed curve
    if len(training_history['episode_rewards']) > 50:
        window_size = min(50, len(training_history['episode_rewards']) // 10)
        smoothed = np.convolve(training_history['episode_rewards'], np.ones(window_size)/window_size, mode='valid')
        x_smooth = np.arange(window_size-1, len(training_history['episode_rewards']))
        plt.plot(x_smooth, smoothed, 'r-', linewidth=2, label='Smoothed')
    plt.title(f'Episode Rewards - {config_name.upper()}\n({num_agents} agents, {num_episodes} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    if len(training_history['episode_rewards']) > 50:
        plt.legend()
    
    plt.subplot(1, 2, 2)
    for key, losses in training_history['episode_losses'].items():
        if losses:
            plt.plot(losses, label=key, alpha=0.7, linewidth=1)
    plt.title(f'Training Losses - {config_name.upper()}')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with config-specific filename
    image_filename = f'training_curves_{config_name}.png'
    plt.savefig(output_dir / image_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save a summary plot with more details
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Rewards
    plt.subplot(2, 2, 1)
    plt.plot(training_history['episode_rewards'], alpha=0.3, color='blue', label='Raw')
    if len(training_history['episode_rewards']) > 50:
        window_size = min(50, len(training_history['episode_rewards']) // 10)
        smoothed = np.convolve(training_history['episode_rewards'], np.ones(window_size)/window_size, mode='valid')
        x_smooth = np.arange(window_size-1, len(training_history['episode_rewards']))
        plt.plot(x_smooth, smoothed, 'r-', linewidth=2, label='Smoothed')
    plt.title(f'Episode Rewards\nConfig: {config_name}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Losses
    plt.subplot(2, 2, 2)
    for key, losses in training_history['episode_losses'].items():
        if losses:
            plt.plot(losses, label=key, alpha=0.7)
    plt.title('Training Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Reward statistics
    plt.subplot(2, 2, 3)
    rewards_array = np.array(training_history['episode_rewards'])
    if len(rewards_array) > 0:
        # Split into segments for analysis
        num_segments = 10
        segment_size = len(rewards_array) // num_segments
        segment_means = []
        segment_stds = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(rewards_array)
            segment = rewards_array[start_idx:end_idx]
            segment_means.append(np.mean(segment))
            segment_stds.append(np.std(segment))
        
        x_segments = np.arange(num_segments) * segment_size + segment_size // 2
        plt.errorbar(x_segments, segment_means, yerr=segment_stds, 
                    fmt='o-', capsize=5, capthick=2, linewidth=2)
        plt.title('Reward Statistics (by Segment)')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward ± Std')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Configuration info
    plt.subplot(2, 2, 4)
    plt.axis('off')
    config_text = f"""
Configuration: {config_name.upper()}

Agents:
  Intelligence: {config['agents']['num_intelligence_agents']}
  Collectors: {config['agents']['num_data_collectors']}
  Generators: {config['agents']['num_insight_generators']}
  Total: {num_agents}

Environment:
  Data Sources: {config['environment']['num_data_sources']}
  Task Types: {config['environment']['num_task_types']}

Training:
  Episodes: {num_episodes}
  Learning Rate: {config['training']['learning_rate']}
  Batch Size: {config['training']['batch_size']}

Results:
  Final Reward: {training_history['episode_rewards'][-1]:.2f}
  Avg Reward: {np.mean(training_history['episode_rewards']):.2f}
  Max Reward: {np.max(training_history['episode_rewards']):.2f}
  Min Reward: {np.min(training_history['episode_rewards']):.2f}
    """
    plt.text(0.1, 0.5, config_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save detailed summary plot
    summary_filename = f'training_summary_{config_name}.png'
    plt.savefig(output_dir / summary_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining complete! Models saved to {output_dir}")
    print(f"Training curves saved to {output_dir / image_filename}")
    print(f"Training summary saved to {output_dir / summary_filename}")


if __name__ == '__main__':
    main()

