"""
Data Collector Agent - Specialized agent for gathering data from various sources
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .base_agent import BaseAgent
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from rl.exploration.ucb import UCBExploration


class DataCollector(BaseAgent):
    """Agent specialized in collecting data from various sources"""
    
    def __init__(
        self,
        agent_id: str,
        source_preferences: List[str],
        collection_rate: float = 1.0,
        memory_size: int = 100
    ):
        """
        Initialize Data Collector
        
        Args:
            agent_id: Unique identifier
            source_preferences: Preferred data source types
            collection_rate: Rate at which agent can collect data
        """
        capabilities = ['data_collection', 'source_evaluation', 'quality_assessment']
        super().__init__(agent_id, 'collector', capabilities, memory_size)
        
        self.source_preferences = source_preferences
        self.collection_rate = collection_rate
        self.collected_data = []
        self.source_quality_history = {}
        
        # UCB exploration for data source selection
        self.ucb_explorer = UCBExploration(c=2.0)
        
    def process_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Process observation into state vector"""
        state_features = []
        
        # Agent state
        state_features.append(len(self.collected_data))
        state_features.append(self.collection_rate)
        
        # Available sources
        if 'available_sources' in observation:
            state_features.append(len(observation['available_sources']))
            
            # Source quality information
            avg_quality = 0.0
            if observation['available_sources']:
                qualities = [s.get('quality', 0.5) for s in observation['available_sources']]
                avg_quality = np.mean(qualities) if qualities else 0.5
            state_features.append(avg_quality)
        else:
            state_features.extend([0, 0.5])
            
        # Data demand
        if 'data_demand' in observation:
            state_features.append(observation['data_demand'])
        else:
            state_features.append(0)
            
        # Source preference match
        if 'available_sources' in observation:
            preferred_count = sum(
                1 for s in observation['available_sources']
                if s.get('type') in self.source_preferences
            )
            state_features.append(preferred_count / max(len(observation['available_sources']), 1))
        else:
            state_features.append(0.0)
            
        return np.array(state_features, dtype=np.float32)
    
    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select action - will be overridden by RL policy"""
        state = self.process_observation(observation)
        
        if not hasattr(self, 'rl_policy'):
            action = self._heuristic_action(observation)
            # Store source index for UCB update
            if 'available_sources' in observation and observation['available_sources']:
                source_ids = [s.get('id') for s in observation['available_sources']]
                if action.get('source_id') in source_ids:
                    self._last_source_index = source_ids.index(action['source_id'])
            return action
        
        return self.rl_policy(state, observation)
    
    def _heuristic_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Heuristic action selection with UCB exploration"""
        action = {
            'type': 'collect',
            'source_id': None,
            'priority': 'medium'
        }
        
        if 'available_sources' in observation and observation['available_sources']:
            # Use UCB for exploration-exploitation trade-off
            source_indices = list(range(len(observation['available_sources'])))
            selected_idx = self.ucb_explorer.select_action(source_indices)
            
            if 0 <= selected_idx < len(observation['available_sources']):
                selected_source = observation['available_sources'][selected_idx]
                action['source_id'] = selected_source['id']
                
                # Determine priority based on quality
                quality = selected_source.get('quality', 0.5)
                action['priority'] = 'high' if quality > 0.7 else 'medium'
        
        return action
    
    def collect_data(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from a source"""
        data = {
            'id': f"data_{self.agent_id}_{len(self.collected_data)}",
            'source_id': source.get('id'),
            'source_type': source.get('type'),
            'quality': source.get('quality', 0.5),
            'timestamp': len(self.collected_data),
            'collector_id': self.agent_id
        }
        
        self.collected_data.append(data)
        
        # Update source quality history
        source_id = source.get('id')
        if source_id not in self.source_quality_history:
            self.source_quality_history[source_id] = []
        self.source_quality_history[source_id].append(data['quality'])
        
        # Update UCB explorer with reward (quality as reward)
        # Find source index for UCB update
        if hasattr(self, '_last_source_index'):
            self.ucb_explorer.update(self._last_source_index, data['quality'])
        
        return data
    
    def evaluate_source_quality(self, source_id: str) -> float:
        """Evaluate quality of a source based on history"""
        if source_id in self.source_quality_history:
            return np.mean(self.source_quality_history[source_id])
        return 0.5
    
    def reset_episode(self):
        """Reset for new episode"""
        super().reset_episode()
        self.collected_data = []

