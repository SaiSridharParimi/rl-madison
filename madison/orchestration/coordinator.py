"""
Agent Coordinator - Orchestrates multi-agent system with RL
"""

from typing import Dict, List, Any, Optional
import numpy as np
from ..agents.base_agent import BaseAgent
from ..agents.intelligence_agent import IntelligenceAgent
from ..agents.data_collector import DataCollector
from ..agents.insight_generator import InsightGenerator
from .task_allocator import TaskAllocator


class AgentCoordinator:
    """
    Coordinates multiple agents in the Madison framework
    """
    
    def __init__(
        self,
        num_intelligence_agents: int = 3,
        num_data_collectors: int = 2,
        num_insight_generators: int = 2,
        task_allocator: Optional[TaskAllocator] = None
    ):
        """
        Initialize coordinator
        
        Args:
            num_intelligence_agents: Number of intelligence agents
            num_data_collectors: Number of data collector agents
            num_insight_generators: Number of insight generator agents
            task_allocator: Task allocator instance
        """
        self.agents: List[BaseAgent] = []
        self.task_allocator = task_allocator or TaskAllocator()
        
        # Create agents
        for i in range(num_intelligence_agents):
            agent = IntelligenceAgent(
                agent_id=f"intelligence_{i}",
                expertise_domains=['marketing', 'analytics', 'strategy'],
                analysis_capacity=5
            )
            self.agents.append(agent)
        
        for i in range(num_data_collectors):
            agent = DataCollector(
                agent_id=f"collector_{i}",
                source_preferences=['api', 'database', 'web'],
                collection_rate=1.0
            )
            self.agents.append(agent)
        
        for i in range(num_insight_generators):
            agent = InsightGenerator(
                agent_id=f"generator_{i}",
                synthesis_style='comprehensive'
            )
            self.agents.append(agent)
        
        self.episode_count = 0
        self.total_reward = 0.0
        
    def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        return [agent for agent in self.agents if agent.agent_type == agent_type]
    
    def coordinate_step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate one step of agent actions
        
        Args:
            observation: Environment observation
            
        Returns:
            Dictionary of agent actions
        """
        actions = {}
        
        # Get pending tasks
        pending_tasks = observation.get('pending_tasks', [])
        
        # Allocate tasks to agents
        for task in pending_tasks:
            # Determine which agent type should handle this task
            task_type = task.get('type', 'general')
            
            if task_type == 'data_collection':
                available_agents = self.get_agents_by_type('collector')
            elif task_type == 'analysis':
                available_agents = self.get_agents_by_type('intelligence')
            elif task_type == 'synthesis':
                available_agents = self.get_agents_by_type('generator')
            else:
                available_agents = self.agents
            
            # Filter available agents (not at capacity)
            available_agents = [
                agent for agent in available_agents
                if self._is_agent_available(agent, task)
            ]
            
            if available_agents:
                allocated_agent = self.task_allocator.allocate_task(
                    task, available_agents, observation
                )
                
                if allocated_agent:
                    # Agent selects action
                    agent_observation = self._prepare_agent_observation(
                        allocated_agent, observation, task
                    )
                    action = allocated_agent.select_action(agent_observation)
                    actions[allocated_agent.agent_id] = action
        
        # Get actions from other agents (not allocated tasks)
        for agent in self.agents:
            if agent.agent_id not in actions:
                agent_observation = self._prepare_agent_observation(agent, observation)
                actions[agent.agent_id] = agent.select_action(agent_observation)
        
        return actions
    
    def _is_agent_available(self, agent: BaseAgent, task: Dict[str, Any]) -> bool:
        """Check if agent is available for task"""
        if isinstance(agent, IntelligenceAgent):
            return len(agent.current_analyses) < agent.analysis_capacity
        elif isinstance(agent, DataCollector):
            return True  # Collectors can always collect
        elif isinstance(agent, InsightGenerator):
            return len(agent.synthesis_queue) < 5  # Reasonable limit
        return True
    
    def _prepare_agent_observation(
        self,
        agent: BaseAgent,
        observation: Dict[str, Any],
        task: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare observation for specific agent"""
        agent_obs = observation.copy()
        
        if task:
            agent_obs['current_task'] = task
            agent_obs['task_domain'] = task.get('domain', 'general')
        
        # Add agent-specific information
        if isinstance(agent, IntelligenceAgent):
            agent_obs['available_data'] = observation.get('collected_data', [])
            agent_obs['pending_tasks'] = [
                t for t in observation.get('pending_tasks', [])
                if t.get('type') == 'analysis'
            ]
        elif isinstance(agent, DataCollector):
            agent_obs['available_sources'] = observation.get('data_sources', [])
            agent_obs['data_demand'] = len(observation.get('pending_tasks', []))
        elif isinstance(agent, InsightGenerator):
            agent_obs['available_analyses'] = observation.get('completed_analyses', [])
            agent_obs['insight_demand'] = observation.get('insight_demand', 0)
        
        return agent_obs
    
    def update_agents(self, rewards: Dict[str, float], done: bool = False):
        """Update agents with rewards"""
        for agent_id, reward in rewards.items():
            agent = self.get_agent_by_id(agent_id)
            if agent:
                agent.receive_reward(reward)
                self.total_reward += reward
        
        if done:
            self.episode_count += 1
            for agent in self.agents:
                agent.reset_episode()
    
    def reset(self):
        """Reset coordinator for new episode"""
        for agent in self.agents:
            agent.reset_episode()
        self.total_reward = 0.0

