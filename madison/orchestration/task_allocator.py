"""
Task Allocator - RL-driven dynamic task allocation system
"""

from typing import Dict, List, Any, Optional
import numpy as np
from ..agents.base_agent import BaseAgent


class TaskAllocator:
    """
    Task allocator that uses RL to learn optimal task allocation strategies
    """
    
    def __init__(self, rl_agent=None):
        """
        Initialize task allocator
        
        Args:
            rl_agent: RL agent for learning allocation strategies (Q-Learning or DQN)
        """
        self.rl_agent = rl_agent
        self.allocation_history = []
        self.performance_metrics = {}
        
    def allocate_task(
        self,
        task: Dict[str, Any],
        available_agents: List[BaseAgent],
        observation: Dict[str, Any]
    ) -> Optional[BaseAgent]:
        """
        Allocate a task to an agent using RL policy
        
        Args:
            task: Task to allocate
            available_agents: List of available agents
            observation: Current environment observation
            
        Returns:
            Selected agent or None if no allocation possible
        """
        if not available_agents:
            return None
        
        if self.rl_agent is None:
            # Heuristic allocation
            return self._heuristic_allocation(task, available_agents)
        
        # RL-based allocation
        state = self._get_allocation_state(task, available_agents, observation)
        action = self.rl_agent.select_action(state, training=True)
        
        if 0 <= action < len(available_agents):
            selected_agent = available_agents[action]
            self.allocation_history.append({
                'task_id': task.get('id'),
                'agent_id': selected_agent.agent_id,
                'action': action
            })
            return selected_agent
        
        return None
    
    def _get_allocation_state(
        self,
        task: Dict[str, Any],
        agents: List[BaseAgent],
        observation: Dict[str, Any]
    ) -> np.ndarray:
        """Extract state representation for allocation decision"""
        state_features = []
        
        # Task features
        task_priority = task.get('priority', 'medium')
        priority_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
        state_features.append(priority_map.get(task_priority, 0.5))
        
        task_complexity = task.get('complexity', 0.5)
        state_features.append(task_complexity)
        
        # Agent features
        state_features.append(len(agents))
        
        # Agent capabilities match
        task_requirements = task.get('requirements', [])
        if task_requirements:
            match_scores = []
            for agent in agents:
                match = sum(1 for req in task_requirements if req in agent.capabilities)
                match_scores.append(match / len(task_requirements))
            state_features.extend(match_scores[:5])  # Top 5 agents
        else:
            state_features.extend([0.0] * 5)
        
        # Agent workload
        workload_scores = []
        for agent in agents:
            if hasattr(agent, 'current_analyses'):
                workload = len(agent.current_analyses) / getattr(agent, 'analysis_capacity', 5)
            elif hasattr(agent, 'collected_data'):
                workload = len(agent.collected_data) / 10.0
            else:
                workload = 0.0
            workload_scores.append(workload)
        state_features.extend(workload_scores[:5])
        
        # Pad or truncate to fixed size
        target_size = 15
        if len(state_features) < target_size:
            state_features.extend([0.0] * (target_size - len(state_features)))
        else:
            state_features = state_features[:target_size]
        
        return np.array(state_features, dtype=np.float32)
    
    def _heuristic_allocation(
        self,
        task: Dict[str, Any],
        agents: List[BaseAgent]
    ) -> Optional[BaseAgent]:
        """Heuristic task allocation (fallback)"""
        if not agents:
            return None
        
        # Select agent with matching capabilities and lowest workload
        best_agent = None
        best_score = -1
        
        task_requirements = task.get('requirements', [])
        
        for agent in agents:
            score = 0.0
            
            # Capability match
            if task_requirements:
                match = sum(1 for req in task_requirements if req in agent.capabilities)
                score += match / len(task_requirements)
            
            # Workload (prefer less loaded agents)
            if hasattr(agent, 'current_analyses'):
                workload = len(agent.current_analyses) / getattr(agent, 'analysis_capacity', 5)
                score += (1.0 - workload) * 0.5
            elif hasattr(agent, 'collected_data'):
                workload = min(len(agent.collected_data) / 10.0, 1.0)
                score += (1.0 - workload) * 0.5
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def update_allocation_performance(
        self,
        task_id: str,
        agent_id: str,
        performance: float
    ):
        """Update performance metrics for allocation"""
        key = f"{task_id}_{agent_id}"
        if key not in self.performance_metrics:
            self.performance_metrics[key] = []
        self.performance_metrics[key].append(performance)
    
    def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get statistics about task allocations"""
        if not self.allocation_history:
            return {}
        
        agent_counts = {}
        for allocation in self.allocation_history:
            agent_id = allocation['agent_id']
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        
        return {
            'total_allocations': len(self.allocation_history),
            'agent_distribution': agent_counts,
            'avg_performance': np.mean([
                np.mean(perfs) for perfs in self.performance_metrics.values()
            ]) if self.performance_metrics else 0.0
        }

