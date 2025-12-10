"""
Base Agent class for Madison framework
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class BaseAgent(ABC):
    """Base class for all agents in the Madison framework"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        memory_size: int = 100
    ):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., 'intelligence', 'collector', 'generator')
            capabilities: List of capabilities this agent has
            memory_size: Size of agent's memory buffer
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.memory = []
        self.memory_size = memory_size
        self.state = {}
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        
    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> Any:
        """Select an action based on current observation"""
        pass
    
    @abstractmethod
    def process_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Process raw observation into state representation"""
        pass
    
    def update_memory(self, experience: Dict[str, Any]):
        """Update agent's memory with new experience"""
        self.memory.append(experience)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state,
            'total_reward': self.total_reward,
            'step_count': self.step_count,
            'memory_size': len(self.memory)
        }
    
    def reset_episode(self):
        """Reset agent state for new episode"""
        self.episode_reward = 0.0
        self.state = {}
    
    def receive_reward(self, reward: float):
        """Receive and accumulate reward"""
        self.episode_reward += reward
        self.total_reward += reward
    
    def communicate(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Base communication method - can be overridden"""
        return {'status': 'received', 'agent_id': self.agent_id}

