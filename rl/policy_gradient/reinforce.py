"""
REINFORCE (Monte Carlo Policy Gradient) implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional

from ..utils.networks import PolicyNetwork


class REINFORCE:
    """
    REINFORCE algorithm for policy gradient learning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        hidden_dims: list = [128, 128, 64],
        device: Optional[torch.device] = None
    ):
        """
        Initialize REINFORCE agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            hidden_dims: Hidden layer dimensions
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Policy network
        self.policy_network = PolicyNetwork(
            state_dim, action_dim, hidden_dims, continuous_action=False
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Storage for episode
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode storage"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """
        Select action using current policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        action_probs, _ = self.policy_network(state_tensor)
        
        if training:
            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # Greedy action
            action = torch.argmax(action_probs, dim=1)
            log_prob = torch.log(action_probs[0, action.item()] + 1e-8)
        
        return action.item(), log_prob.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float
    ):
        """
        Store transition in episode
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
        """
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
    
    def train_step(self) -> float:
        """
        Perform REINFORCE training step using episode returns
        
        Returns:
            Loss value
        """
        if len(self.episode_states) == 0:
            return 0.0
        
        # Compute discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.episode_log_probs).to(self.device)
        
        # Get current policy log probs
        action_probs, _ = self.policy_network(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # REINFORCE loss: -log_prob * return
        loss = -(log_probs * returns).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Reset episode
        self.reset_episode()
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model to file"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

