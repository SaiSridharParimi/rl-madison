"""
Proximal Policy Optimization (PPO) implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque

from ..utils.networks import PolicyNetwork, ValueNetwork


class PPOAgent:
    """
    Proximal Policy Optimization agent
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        hidden_dims: list = [128, 128, 64],
        device: Optional[torch.device] = None
    ):
        """
        Initialize PPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs
            hidden_dims: Hidden layer dimensions
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Networks
        self.policy_network = PolicyNetwork(
            state_dim, action_dim, hidden_dims, continuous_action=False
        ).to(self.device)
        
        self.value_network = ValueNetwork(state_dim, hidden_dims).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=learning_rate
        )
        
        # Storage for trajectories
        self.reset_storage()
    
    def reset_storage(self):
        """Reset trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select action using current policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set to eval mode for single-sample inference to avoid BatchNorm issues
        was_policy_training = self.policy_network.training
        was_value_training = self.value_network.training
        self.policy_network.eval()
        self.value_network.eval()
        
        with torch.no_grad():
            action_probs, action_logits = self.policy_network(state_tensor)
            value = self.value_network(state_tensor).squeeze()
            
            if training:
                # Sample action
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                # Greedy action
                action = torch.argmax(action_probs, dim=1)
                log_prob = torch.log(action_probs[0, action.item()] + 1e-8)
        
        # Restore training mode
        if was_policy_training:
            self.policy_network.train()
        if was_value_training:
            self.value_network.train()
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """
        Store transition in trajectory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: State value estimate
            done: Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            next_value: Value estimate for terminal state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        gae = 0
        
        values = self.values + [next_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform PPO training step
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.states) == 0:
            return {}
        
        # Compute advantages and returns
        next_value = 0.0 if self.dones[-1] else self.value_network(
            torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
        ).item()
        
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.ppo_epochs):
            # Get current policy predictions
            action_probs, action_logits = self.policy_network(states)
            values = self.value_network(states).squeeze()
            
            # Policy loss
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Importance sampling ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns_tensor)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Reset storage
        self.reset_storage()
        
        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs
        }
    
    def save(self, filepath: str):
        """Save model to file"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

