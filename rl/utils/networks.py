"""
Neural network architectures for RL algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DQNNetwork(nn.Module):
    """Deep Q-Network for value-based learning"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128, 64],
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        """
        Initialize DQN network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            use_batch_norm: Whether to use batch normalization
        """
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            # BatchNorm disabled for single-sample inference compatibility
            # BatchNorm requires batch_size > 1 in training mode
            # if use_batch_norm:
            #     layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)


class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128, 64],
        activation: str = "relu",
        use_batch_norm: bool = True,
        continuous_action: bool = False
    ):
        """
        Initialize policy network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            continuous_action: Whether actions are continuous
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_action = continuous_action
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Note: BatchNorm disabled for single-sample inference compatibility
            # Can be enabled for batch training, but requires batch_size > 1
            # if use_batch_norm:
            #     layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        if continuous_action:
            # For continuous actions, output mean and std
            self.mean_layer = nn.Linear(input_dim, action_dim)
            self.std_layer = nn.Linear(input_dim, action_dim)
        else:
            # For discrete actions, output action probabilities
            self.action_layer = nn.Linear(input_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            action_probs or (mean, std) for continuous actions
            log_probs
        """
        features = self.shared_layers(state)
        
        if self.continuous_action:
            mean = self.mean_layer(features)
            std = F.softplus(self.std_layer(features)) + 1e-5
            return mean, std
        else:
            action_logits = self.action_layer(features)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs, action_logits


class ValueNetwork(nn.Module):
    """Value network for policy gradient methods"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [128, 128, 64],
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        """
        Initialize value network
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
        """
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            # Note: BatchNorm disabled for single-sample inference compatibility
            # if use_batch_norm:
            #     layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim
        
        # Output layer (single value)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)

