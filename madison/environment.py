"""
Madison Environment - Simulated environment for RL training
"""

import numpy as np
from typing import Dict, List, Any, Optional
import random
from .orchestration.coordinator import AgentCoordinator
from .tools.data_tools import DataSource, DataQualityAssessor
from .tools.analysis_tools import AnalysisEngine, InsightSynthesizer


class MadisonEnvironment:
    """
    Environment for training RL agents in Madison framework
    """
    
    def __init__(
        self,
        num_data_sources: int = 10,
        num_task_types: int = 5,
        coordinator: Optional[AgentCoordinator] = None
    ):
        """
        Initialize environment
        
        Args:
            num_data_sources: Number of data sources
            num_task_types: Number of different task types
            coordinator: Agent coordinator instance
        """
        self.num_data_sources = num_data_sources
        self.num_task_types = num_task_types
        self.coordinator = coordinator
        
        # Initialize data sources
        self.data_sources = []
        source_types = ['api', 'database', 'web', 'file', 'stream']
        for i in range(num_data_sources):
            source = DataSource(
                source_id=f"source_{i}",
                source_type=random.choice(source_types),
                quality=np.random.uniform(0.5, 0.9),
                availability=np.random.uniform(0.7, 1.0)
            )
            self.data_sources.append(source)
        
        # State tracking
        self.collected_data = []
        self.completed_analyses = []
        self.generated_insights = []
        self.pending_tasks = []
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Tools
        self.quality_assessor = DataQualityAssessor()
        self.analysis_engine = AnalysisEngine()
        self.insight_synthesizer = InsightSynthesizer()
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment for new episode"""
        self.collected_data = []
        self.completed_analyses = []
        self.generated_insights = []
        self.pending_tasks = []
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Generate initial tasks
        self._generate_tasks(num_tasks=5)
        
        return self.get_observation()
    
    def step(self, actions: Dict[str, Any]) -> tuple:
        """
        Execute one environment step
        
        Args:
            actions: Dictionary of agent actions
            
        Returns:
            Tuple of (observation, rewards, done, info)
        """
        self.step_count += 1
        rewards = {}
        
        # Process actions
        for agent_id, action in actions.items():
            reward = self._process_action(agent_id, action)
            rewards[agent_id] = reward
        
        # Generate new tasks periodically
        if self.step_count % 5 == 0:
            self._generate_tasks(num_tasks=random.randint(1, 3))
        
        # Calculate overall reward
        total_reward = sum(rewards.values())
        self.episode_reward += total_reward
        
        # Check if done
        done = self.step_count >= 100 or len(self.generated_insights) >= 10
        
        observation = self.get_observation()
        info = {
            'step': self.step_count,
            'total_reward': self.episode_reward,
            'num_insights': len(self.generated_insights),
            'num_analyses': len(self.completed_analyses),
            'num_data': len(self.collected_data)
        }
        
        return observation, rewards, done, info
    
    def _process_action(self, agent_id: str, action: Dict[str, Any]) -> float:
        """Process an agent action and return reward with enhanced reward engineering"""
        action_type = action.get('type', 'unknown')
        reward = 0.0
        
        try:
            if action_type == 'collect':
                # Data collection action
                source_id = action.get('source_id')
                if source_id:
                    source = next((s for s in self.data_sources if s.source_id == source_id), None)
                    if source:
                        data = source.collect()
                        if data.get('status') != 'unavailable':
                            self.collected_data.append(data)
                            # Enhanced reward: quality + diversity bonus
                            quality = data.get('quality', 0.5)
                            base_reward = quality * 10.0
                            
                            # Diversity bonus: reward exploring new sources
                            source_types = [d.get('source_type') for d in self.collected_data[-10:]]
                            if source_types.count(data.get('source_type', '')) == 1:
                                base_reward += 2.0  # Diversity bonus
                            
                            reward = base_reward
                        else:
                            reward = -1.0  # Source unavailable
                    else:
                        reward = -2.0  # Invalid source
                else:
                    reward = -1.0  # No source selected
                    
            elif action_type == 'analyze':
                # Analysis action
                target = action.get('target')
                if self.collected_data:
                    # Use available data for analysis
                    data = self.collected_data[-1] if not target else next(
                        (d for d in self.collected_data if d.get('id') == target), None
                    )
                    if data:
                        analysis = self.analysis_engine.analyze_data(data, 'marketing')
                        self.completed_analyses.append(analysis)
                        # Enhanced reward: completeness + timeliness
                        completeness = analysis.get('completeness', 0.5)
                        base_reward = completeness * 15.0
                        
                        # Timeliness bonus: analyze fresh data
                        if len(self.collected_data) - self.collected_data.index(data) <= 2:
                            base_reward += 3.0
                        
                        reward = base_reward
                    else:
                        reward = -1.0  # Data not found
                else:
                    reward = -0.5  # No data available
                    
            elif action_type == 'synthesize':
                # Synthesis action
                analysis_ids = action.get('analysis_ids', [])
                if analysis_ids and self.completed_analyses:
                    analyses = [
                        a for a in self.completed_analyses
                        if a.get('id') in analysis_ids
                    ]
                    if analyses:
                        insight = self.insight_synthesizer.synthesize(analyses, 'comprehensive')
                        self.generated_insights.append(insight)
                        # Enhanced reward: quality + synthesis complexity
                        quality = insight.get('quality', 0.5)
                        base_reward = quality * 20.0
                        
                        # Complexity bonus: synthesizing more analyses
                        complexity_bonus = min(len(analyses) * 2.0, 10.0)
                        reward = base_reward + complexity_bonus
                    else:
                        reward = -1.0  # Analyses not found
                else:
                    reward = -0.5  # No analyses available
            
            else:
                reward = -0.1  # Unknown action type
            
            # Priority bonus
            if action.get('priority') == 'high':
                reward *= 1.2
            
            # Efficiency bonus: reward agents for completing workflows
            if len(self.generated_insights) > 0 and len(self.completed_analyses) > 0:
                workflow_efficiency = len(self.generated_insights) / max(len(self.completed_analyses), 1)
                if workflow_efficiency > 0.5:
                    reward += 1.0
        
        except Exception as e:
            # Error handling: penalize invalid actions
            reward = -5.0
        
        return reward
    
    def _generate_tasks(self, num_tasks: int = 1):
        """Generate new tasks"""
        task_types = ['data_collection', 'analysis', 'synthesis']
        priorities = ['low', 'medium', 'high']
        
        for _ in range(num_tasks):
            task = {
                'id': f"task_{self.step_count}_{len(self.pending_tasks)}",
                'type': random.choice(task_types),
                'priority': random.choice(priorities),
                'complexity': np.random.uniform(0.3, 0.9),
                'domain': random.choice(['marketing', 'analytics', 'strategy']),
                'requirements': random.sample(['data_collection', 'data_analysis', 'insight_generation'], k=random.randint(1, 2))
            }
            self.pending_tasks.append(task)
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current environment observation"""
        return {
            'data_sources': [s.get_info() for s in self.data_sources],
            'collected_data': self.collected_data[-10:],  # Last 10 items
            'completed_analyses': self.completed_analyses[-10:],
            'generated_insights': self.generated_insights,
            'pending_tasks': self.pending_tasks,
            'step': self.step_count,
            'data_quality': np.mean([d.get('quality', 0.5) for d in self.collected_data]) if self.collected_data else 0.5,
            'insight_demand': max(0, 5 - len(self.generated_insights))
        }
    
    def get_reward_components(self) -> Dict[str, float]:
        """Get breakdown of reward components"""
        return {
            'data_collection': len(self.collected_data) * 0.1,
            'analysis': len(self.completed_analyses) * 0.15,
            'insight_generation': len(self.generated_insights) * 0.2,
            'quality_bonus': np.mean([i.get('quality', 0.5) for i in self.generated_insights]) * 10 if self.generated_insights else 0.0
        }

