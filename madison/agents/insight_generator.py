"""
Insight Generator Agent - Specialized agent for synthesizing insights from analyses
"""

from typing import Dict, Any, List
import numpy as np
from .base_agent import BaseAgent


class InsightGenerator(BaseAgent):
    """Agent specialized in generating and synthesizing marketing insights"""
    
    def __init__(
        self,
        agent_id: str,
        synthesis_style: str = 'comprehensive',
        memory_size: int = 100
    ):
        """
        Initialize Insight Generator
        
        Args:
            agent_id: Unique identifier
            synthesis_style: Style of synthesis ('comprehensive', 'concise', 'strategic')
        """
        capabilities = ['insight_synthesis', 'report_generation', 'recommendation']
        super().__init__(agent_id, 'generator', capabilities, memory_size)
        
        self.synthesis_style = synthesis_style
        self.generated_reports = []
        self.synthesis_queue = []
        
    def process_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Process observation into state vector"""
        state_features = []
        
        # Agent state
        state_features.append(len(self.synthesis_queue))
        state_features.append(len(self.generated_reports))
        
        # Available analyses
        if 'available_analyses' in observation:
            state_features.append(len(observation['available_analyses']))
            
            # Analysis quality/completeness
            if observation['available_analyses']:
                completeness = np.mean([
                    a.get('completeness', 0.5) for a in observation['available_analyses']
                ])
                state_features.append(completeness)
            else:
                state_features.append(0.0)
        else:
            state_features.extend([0, 0.0])
            
        # Insight demand
        if 'insight_demand' in observation:
            state_features.append(observation['insight_demand'])
        else:
            state_features.append(0)
            
        return np.array(state_features, dtype=np.float32)
    
    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select action - will be overridden by RL policy"""
        state = self.process_observation(observation)
        
        if not hasattr(self, 'rl_policy'):
            return self._heuristic_action(observation)
        
        return self.rl_policy(state, observation)
    
    def _heuristic_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Heuristic action selection"""
        action = {
            'type': 'synthesize',
            'analysis_ids': [],
            'priority': 'medium'
        }
        
        if 'available_analyses' in observation and observation['available_analyses']:
            # Select most complete analyses
            sorted_analyses = sorted(
                observation['available_analyses'],
                key=lambda x: x.get('completeness', 0.5),
                reverse=True
            )
            
            # Select top analyses for synthesis
            num_to_synthesize = min(3, len(sorted_analyses))
            action['analysis_ids'] = [a['id'] for a in sorted_analyses[:num_to_synthesize]]
            action['priority'] = 'high' if num_to_synthesize > 0 else 'medium'
        
        return action
    
    def synthesize_insight(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize insights from multiple analyses"""
        if not analyses:
            return {'status': 'no_analyses', 'agent_id': self.agent_id}
        
        # Calculate synthesis quality based on input analyses
        avg_completeness = np.mean([a.get('completeness', 0.5) for a in analyses])
        num_analyses = len(analyses)
        
        insight = {
            'id': f"insight_{self.agent_id}_{len(self.generated_reports)}",
            'analysis_ids': [a.get('id') for a in analyses],
            'synthesis_style': self.synthesis_style,
            'quality': min(0.95, avg_completeness + 0.1 * min(num_analyses, 3)),
            'content': f"Synthesized insight from {num_analyses} analyses",
            'recommendations': self._generate_recommendations(analyses),
            'agent_id': self.agent_id
        }
        
        self.generated_reports.append(insight)
        return insight
    
    def _generate_recommendations(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analyses"""
        recommendations = []
        
        if analyses:
            recommendations.append("Consider expanding data collection in high-performing areas")
            recommendations.append("Monitor trends identified in recent analyses")
            
        return recommendations
    
    def reset_episode(self):
        """Reset for new episode"""
        super().reset_episode()
        self.synthesis_queue = []

