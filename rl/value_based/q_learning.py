"""
Q-Learning implementation for discrete action spaces
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle


class QLearning:
    """
    Q-Learning algorithm for value-based reinforcement learning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize Q-Learning agent
        
        Args:
            state_dim: Dimension of state space (for discrete states, use state_size)
            action_dim: Number of possible actions
            learning_rate: Learning rate for Q-value updates
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum exploration rate
        """
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        # For continuous states, we'll use a discretization function
        self.q_table = {}
        self.state_discretizer = None
        
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize continuous state to tuple for Q-table lookup
        
        Args:
            state: Continuous state vector
            
        Returns:
            Discretized state tuple
        """
        # Simple discretization: round to 2 decimal places
        if self.state_discretizer is None:
            # Initialize discretization bins
            self.state_bins = [10] * len(state)  # 10 bins per dimension
            self.state_ranges = [
                (state[i] - 1.0, state[i] + 1.0) for i in range(len(state))
            ]
        
        discretized = tuple(
            int(np.clip(
                (state[i] - self.state_ranges[i][0]) / 
                (self.state_ranges[i][1] - self.state_ranges[i][0]) * self.state_bins[i],
                0, self.state_bins[i] - 1
            ))
            for i in range(len(state))
        )
        
        return discretized
    
    def _get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair"""
        state_key = self._discretize_state(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        return self.q_table[state_key][action]
    
    def _set_q_value(self, state: np.ndarray, action: int, value: float):
        """Set Q-value for state-action pair"""
        state_key = self._discretize_state(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        self.q_table[state_key][action] = value
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_dim)
        
        # Exploit: best action according to Q-values
        state_key = self._discretize_state(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        return np.argmax(self.q_table[state_key])
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Update Q-values using Q-Learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        current_q = self._get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            # Q-Learning: use max Q-value of next state
            next_state_key = self._discretize_state(next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_dim)
            
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * max_next_q
        
        # Q-Learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self._set_q_value(state, action, new_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for state (greedy policy)
        
        Args:
            state: Current state
            
        Returns:
            Action probabilities
        """
        state_key = self._discretize_state(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        # Greedy policy: one-hot for best action
        probs = np.zeros(self.action_dim)
        best_action = np.argmax(self.q_table[state_key])
        probs[best_action] = 1.0
        
        return probs
    
    def save(self, filepath: str):
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'state_bins': getattr(self, 'state_bins', None),
                'state_ranges': getattr(self, 'state_ranges', None)
            }, f)
    
    def load(self, filepath: str):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            if 'state_bins' in data:
                self.state_bins = data['state_bins']
            if 'state_ranges' in data:
                self.state_ranges = data['state_ranges']

