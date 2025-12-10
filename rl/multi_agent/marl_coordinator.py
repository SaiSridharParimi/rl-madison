"""
Multi-Agent Reinforcement Learning Coordinator
Implements coordinated learning across agent teams
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .reward_sharing import RewardSharing


class MARLCoordinator:
    """
    Coordinates multi-agent reinforcement learning with reward sharing
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        sharing_strategy: str = 'proportional',
        communication_enabled: bool = True
    ):
        """
        Initialize MARL coordinator
        
        Args:
            agent_ids: List of agent IDs participating in MARL
            sharing_strategy: Reward sharing strategy ('equal', 'proportional', 'competitive', 'team_bonus')
            communication_enabled: Whether agents can communicate
        """
        self.agent_ids = agent_ids
        self.sharing_strategy = sharing_strategy
        self.communication_enabled = communication_enabled
        
        # Track agent performance
        self.agent_performance = {agent_id: [] for agent_id in agent_ids}
        self.agent_contributions = {agent_id: 1.0 for agent_id in agent_ids}
        self.communication_history = []
        
    def share_rewards(
        self,
        individual_rewards: Dict[str, float],
        team_performance: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Share rewards among agents based on strategy
        
        Args:
            individual_rewards: Dictionary of agent_id -> individual reward
            team_performance: Optional overall team performance metric
            
        Returns:
            Dictionary of agent_id -> shared reward
        """
        if self.sharing_strategy == 'equal':
            return RewardSharing.equal_sharing(individual_rewards, self.agent_ids)
        
        elif self.sharing_strategy == 'proportional':
            return RewardSharing.proportional_sharing(
                individual_rewards,
                self.agent_ids,
                self.agent_contributions
            )
        
        elif self.sharing_strategy == 'competitive':
            # Rank agents by performance
            performance_ranking = sorted(
                self.agent_ids,
                key=lambda aid: np.mean(self.agent_performance.get(aid, [0.0])),
                reverse=True
            )
            return RewardSharing.competitive_sharing(
                individual_rewards,
                self.agent_ids,
                performance_ranking
            )
        
        elif self.sharing_strategy == 'team_bonus':
            if team_performance is None:
                team_performance = sum(individual_rewards.values()) / len(individual_rewards)
            return RewardSharing.team_bonus(
                individual_rewards,
                self.agent_ids,
                team_performance
            )
        
        else:
            # Default to equal sharing
            return RewardSharing.equal_sharing(individual_rewards, self.agent_ids)
    
    def update_performance(self, agent_id: str, reward: float):
        """
        Update performance tracking for an agent
        
        Args:
            agent_id: Agent identifier
            reward: Reward received by agent
        """
        if agent_id in self.agent_performance:
            self.agent_performance[agent_id].append(reward)
            # Keep only recent performance (sliding window)
            if len(self.agent_performance[agent_id]) > 100:
                self.agent_performance[agent_id] = self.agent_performance[agent_id][-100:]
    
    def update_contribution(self, agent_id: str, contribution: float):
        """
        Update contribution weight for an agent
        
        Args:
            agent_id: Agent identifier
            contribution: Contribution weight (0.0 to 1.0)
        """
        if agent_id in self.agent_contributions:
            self.agent_contributions[agent_id] = np.clip(contribution, 0.0, 1.0)
    
    def communicate(
        self,
        sender_id: str,
        receiver_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Handle communication between agents
        
        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID
            message: Message content
            
        Returns:
            Whether communication was successful
        """
        if not self.communication_enabled:
            return False
        
        if sender_id not in self.agent_ids or receiver_id not in self.agent_ids:
            return False
        
        self.communication_history.append({
            'sender': sender_id,
            'receiver': receiver_id,
            'message': message,
            'timestamp': len(self.communication_history)
        })
        
        return True
    
    def get_team_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about team performance
        
        Returns:
            Dictionary of team statistics
        """
        stats = {
            'num_agents': len(self.agent_ids),
            'sharing_strategy': self.sharing_strategy,
            'agent_performance': {},
            'agent_contributions': self.agent_contributions.copy(),
            'total_communications': len(self.communication_history)
        }
        
        for agent_id in self.agent_ids:
            perf = self.agent_performance.get(agent_id, [])
            stats['agent_performance'][agent_id] = {
                'mean': np.mean(perf) if perf else 0.0,
                'std': np.std(perf) if len(perf) > 1 else 0.0,
                'count': len(perf)
            }
        
        return stats
    
    def reset(self):
        """Reset coordinator state"""
        self.agent_performance = {agent_id: [] for agent_id in self.agent_ids}
        self.communication_history = []

