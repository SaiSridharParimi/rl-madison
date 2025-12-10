"""
Upper Confidence Bound (UCB) exploration strategy
"""

import numpy as np
from typing import Dict, List, Optional
import math


class UCBExploration:
    """
    Upper Confidence Bound algorithm for exploration-exploitation trade-off
    """
    
    def __init__(self, c: float = 2.0):
        """
        Initialize UCB exploration
        
        Args:
            c: Exploration constant (higher = more exploration)
        """
        self.c = c
        self.action_counts = {}
        self.action_rewards = {}
        self.total_pulls = 0
        
    def select_action(self, action_space: List[int], state: Optional[np.ndarray] = None) -> int:
        """
        Select action using UCB algorithm
        
        Args:
            action_space: List of available action indices
            state: Optional state for contextual UCB
            
        Returns:
            Selected action index
        """
        if not action_space:
            return 0
        
        # Initialize actions if not seen before
        for action in action_space:
            if action not in self.action_counts:
                self.action_counts[action] = 0
                self.action_rewards[action] = []
        
        # Calculate UCB values for each action
        ucb_values = {}
        for action in action_space:
            count = self.action_counts[action]
            
            if count == 0:
                # Never tried this action - high priority
                ucb_values[action] = float('inf')
            else:
                # Calculate average reward
                avg_reward = np.mean(self.action_rewards[action]) if self.action_rewards[action] else 0.0
                
                # Calculate UCB value
                confidence = self.c * math.sqrt(math.log(self.total_pulls + 1) / count)
                ucb_values[action] = avg_reward + confidence
        
        # Select action with highest UCB value
        best_action = max(ucb_values, key=ucb_values.get)
        return best_action
    
    def update(self, action: int, reward: float):
        """
        Update statistics after taking an action
        
        Args:
            action: Action taken
            reward: Reward received
        """
        if action not in self.action_counts:
            self.action_counts[action] = 0
            self.action_rewards[action] = []
        
        self.action_counts[action] += 1
        self.action_rewards[action].append(reward)
        self.total_pulls += 1
        
        # Keep only recent rewards to adapt to non-stationary environments
        if len(self.action_rewards[action]) > 100:
            self.action_rewards[action] = self.action_rewards[action][-100:]
    
    def get_action_statistics(self) -> Dict[int, Dict[str, float]]:
        """
        Get statistics for each action
        
        Returns:
            Dictionary mapping actions to their statistics
        """
        stats = {}
        for action in self.action_counts:
            rewards = self.action_rewards.get(action, [])
            stats[action] = {
                'count': self.action_counts[action],
                'avg_reward': np.mean(rewards) if rewards else 0.0,
                'std_reward': np.std(rewards) if len(rewards) > 1 else 0.0,
                'total_reward': sum(rewards)
            }
        return stats
    
    def reset(self):
        """Reset exploration statistics"""
        self.action_counts = {}
        self.action_rewards = {}
        self.total_pulls = 0

