"""
Contextual Bandit for situation-aware exploration
"""

import numpy as np
from typing import Dict, List, Optional
from .ucb import UCBExploration


class ContextualBandit:
    """
    Contextual bandit that adapts exploration based on context/state
    """
    
    def __init__(self, num_actions: int, context_dim: int, c: float = 2.0):
        """
        Initialize contextual bandit
        
        Args:
            num_actions: Number of possible actions
            context_dim: Dimension of context/state vector
            c: Exploration constant
        """
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.c = c
        
        # Separate UCB for each context cluster
        self.ucb_explorers = {}
        self.context_clusters = {}
        
    def _get_context_key(self, context: np.ndarray) -> str:
        """
        Convert context to a key for clustering
        
        Args:
            context: Context vector
            
        Returns:
            Context key string
        """
        # Simple discretization: round to 2 decimal places and create key
        rounded = np.round(context, decimals=1)
        return str(rounded.tolist())
    
    def select_action(self, context: np.ndarray, available_actions: Optional[List[int]] = None) -> int:
        """
        Select action based on context
        
        Args:
            context: Current context/state vector
            available_actions: List of available action indices (None = all actions)
            
        Returns:
            Selected action index
        """
        if available_actions is None:
            available_actions = list(range(self.num_actions))
        
        context_key = self._get_context_key(context)
        
        # Get or create UCB explorer for this context
        if context_key not in self.ucb_explorers:
            self.ucb_explorers[context_key] = UCBExploration(c=self.c)
            self.context_clusters[context_key] = []
        
        # Select action using UCB for this context
        action = self.ucb_explorers[context_key].select_action(available_actions, context)
        return action
    
    def update(self, context: np.ndarray, action: int, reward: float):
        """
        Update statistics after taking an action in a context
        
        Args:
            context: Context in which action was taken
            action: Action taken
            reward: Reward received
        """
        context_key = self._get_context_key(context)
        
        if context_key not in self.ucb_explorers:
            self.ucb_explorers[context_key] = UCBExploration(c=self.c)
            self.context_clusters[context_key] = []
        
        # Update UCB explorer for this context
        self.ucb_explorers[context_key].update(action, reward)
        self.context_clusters[context_key].append((context, action, reward))
    
    def get_context_statistics(self) -> Dict[str, Dict]:
        """Get statistics for each context cluster"""
        stats = {}
        for context_key, ucb in self.ucb_explorers.items():
            stats[context_key] = {
                'action_stats': ucb.get_action_statistics(),
                'total_samples': len(self.context_clusters.get(context_key, []))
            }
        return stats

