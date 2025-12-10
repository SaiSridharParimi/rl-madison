"""
Deep Q-Network (DQN) implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import random

from ..utils.networks import DQNNetwork
from ..utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        replay_buffer_size: int = 10000,
        target_update_frequency: int = 10,
        hidden_dims: list = [128, 128, 64],
        device: Optional[torch.device] = None
    ):
        """
        Initialize DQN agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum exploration rate
            batch_size: Batch size for training
            replay_buffer_size: Size of experience replay buffer
            target_update_frequency: Frequency of target network updates
            hidden_dims: Hidden layer dimensions
            device: PyTorch device (CPU or CUDA)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.update_counter = 0
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Q-networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # Set to eval mode for single-sample inference
        was_training = self.q_network.training
        self.q_network.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        # Restore training mode
        if was_training:
            self.q_network.train()
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Ensure states have correct dimensions
        if state.shape[0] != self.state_dim:
            # Pad or truncate to correct dimension
            if state.shape[0] < self.state_dim:
                state = np.pad(state, (0, self.state_dim - state.shape[0]), mode='constant')
            else:
                state = state[:self.state_dim]
        
        if next_state.shape[0] != self.state_dim:
            # Pad or truncate to correct dimension
            if next_state.shape[0] < self.state_dim:
                next_state = np.pad(next_state, (0, self.state_dim - next_state.shape[0]), mode='constant')
            else:
                next_state = next_state[:self.state_dim]
        
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Ensure states have correct dimensions
        if len(states) > 0:
            # Check and fix state dimensions
            if states[0].shape[0] != self.state_dim:
                # Filter out invalid states or pad/truncate
                valid_indices = [i for i, s in enumerate(states) if s.shape[0] == self.state_dim]
                if len(valid_indices) < self.batch_size:
                    return None  # Not enough valid samples
                states = states[valid_indices[:self.batch_size]]
                actions = actions[valid_indices[:self.batch_size]]
                rewards = rewards[valid_indices[:self.batch_size]]
                next_states = next_states[valid_indices[:self.batch_size]]
                dones = dones[valid_indices[:self.batch_size]]
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones.float()) * self.gamma * next_q_value
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model to file"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.update_counter = checkpoint.get('update_counter', 0)

