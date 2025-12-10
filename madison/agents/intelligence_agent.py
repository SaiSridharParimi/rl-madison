"""
Intelligence Agent - Specialized agent for data analysis and insight generation
"""

from typing import Dict, Any, List
import numpy as np
from .base_agent import BaseAgent


class IntelligenceAgent(BaseAgent):
    """Agent specialized in analyzing data and generating marketing insights"""
    
    def __init__(
        self,
        agent_id: str,
        expertise_domains: List[str],
        analysis_capacity: int = 5,
        memory_size: int = 100
    ):
        """
        Initialize Intelligence Agent
        
        Args:
            agent_id: Unique identifier
            expertise_domains: Domains this agent specializes in
            analysis_capacity: Maximum number of analyses per step
        """
        capabilities = ['data_analysis', 'insight_generation', 'pattern_recognition']
        super().__init__(agent_id, 'intelligence', capabilities, memory_size)
        
        self.expertise_domains = expertise_domains
        self.analysis_capacity = analysis_capacity
        self.current_analyses = []
        self.insights_generated = []
        
    def process_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Process observation into state vector"""
        # Extract relevant features from observation
        state_features = []
        
        # Agent state
        state_features.append(len(self.current_analyses) / self.analysis_capacity)
        state_features.append(len(self.insights_generated))
        
        # Observation features
        if 'available_data' in observation:
            state_features.append(len(observation['available_data']))
        else:
            state_features.append(0)
            
        if 'pending_tasks' in observation:
            state_features.append(len(observation['pending_tasks']))
        else:
            state_features.append(0)
            
        if 'data_quality' in observation:
            state_features.append(observation['data_quality'])
        else:
            state_features.append(0.5)
            
        # Domain expertise match
        if 'task_domain' in observation:
            domain_match = 1.0 if observation['task_domain'] in self.expertise_domains else 0.0
            state_features.append(domain_match)
        else:
            state_features.append(0.0)
            
        return np.array(state_features, dtype=np.float32)
    
    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select action using RL policy or heuristic
        
        Args:
            observation: Current environment observation
            
        Returns:
            Action dictionary
        """
        state = self.process_observation(observation)
        
        # Use RL policy if available
        if hasattr(self, 'rl_policy') and self.rl_policy is not None:
            # RL policy should return (action_dict, log_prob, value)
            result = self.rl_policy(state, observation)
            if isinstance(result, tuple) and len(result) == 3:
                action_dict, log_prob, value = result
                # Store for training
                self._last_log_prob = log_prob
                self._last_value = value
                return action_dict
            elif isinstance(result, dict):
                return result
        
        # Fallback to heuristic
        return self._heuristic_action(observation)
    
    def _heuristic_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Heuristic action selection (fallback)"""
        action = {
            'type': 'analyze',
            'target': None,
            'priority': 'medium'
        }
        
        if 'pending_tasks' in observation and observation['pending_tasks']:
            # Select task matching expertise
            for task in observation['pending_tasks']:
                if task.get('domain') in self.expertise_domains:
                    action['target'] = task['id']
                    action['priority'] = 'high'
                    break
        
        return action
    
    def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on data"""
        if len(self.current_analyses) >= self.analysis_capacity:
            return {'status': 'capacity_full', 'agent_id': self.agent_id}
        
        # Simulate analysis
        analysis_id = f"analysis_{self.agent_id}_{len(self.current_analyses)}"
        analysis = {
            'id': analysis_id,
            'data_id': data.get('id'),
            'status': 'in_progress',
            'agent_id': self.agent_id
        }
        
        self.current_analyses.append(analysis)
        return analysis
    
    def generate_insight(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insight from analysis result"""
        insight = {
            'id': f"insight_{self.agent_id}_{len(self.insights_generated)}",
            'analysis_id': analysis_result.get('id'),
            'content': f"Insight generated by {self.agent_id}",
            'confidence': np.random.uniform(0.7, 0.95),
            'agent_id': self.agent_id
        }
        
        self.insights_generated.append(insight)
        return insight
    
    def reset_episode(self):
        """Reset for new episode"""
        super().reset_episode()
        self.current_analyses = []
        self.insights_generated = []

