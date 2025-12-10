"""
Reward sharing mechanisms for multi-agent reinforcement learning
"""

from typing import Dict, List, Any
import numpy as np


class RewardSharing:
    """
    Implements various reward sharing strategies for multi-agent learning
    """
    
    @staticmethod
    def equal_sharing(rewards: Dict[str, float], agent_ids: List[str]) -> Dict[str, float]:
        """
        Equal reward sharing - all agents get equal share
        
        Args:
            rewards: Dictionary of agent_id -> reward
            agent_ids: List of all agent IDs
            
        Returns:
            Dictionary of agent_id -> shared reward
        """
        total_reward = sum(rewards.values())
        shared_reward = total_reward / len(agent_ids) if agent_ids else 0.0
        
        return {agent_id: shared_reward for agent_id in agent_ids}
    
    @staticmethod
    def proportional_sharing(
        rewards: Dict[str, float],
        agent_ids: List[str],
        contribution_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Proportional reward sharing based on contribution weights
        
        Args:
            rewards: Dictionary of agent_id -> reward
            agent_ids: List of all agent IDs
            contribution_weights: Dictionary of agent_id -> contribution weight
            
        Returns:
            Dictionary of agent_id -> shared reward
        """
        total_reward = sum(rewards.values())
        total_weight = sum(contribution_weights.get(agent_id, 0.0) for agent_id in agent_ids)
        
        if total_weight == 0:
            return RewardSharing.equal_sharing(rewards, agent_ids)
        
        shared_rewards = {}
        for agent_id in agent_ids:
            weight = contribution_weights.get(agent_id, 0.0)
            shared_rewards[agent_id] = (weight / total_weight) * total_reward
        
        return shared_rewards
    
    @staticmethod
    def competitive_sharing(
        rewards: Dict[str, float],
        agent_ids: List[str],
        performance_ranking: List[str]
    ) -> Dict[str, float]:
        """
        Competitive reward sharing - top performers get more
        
        Args:
            rewards: Dictionary of agent_id -> reward
            agent_ids: List of all agent IDs
            performance_ranking: List of agent IDs ranked by performance (best first)
            
        Returns:
            Dictionary of agent_id -> shared reward
        """
        total_reward = sum(rewards.values())
        num_agents = len(agent_ids)
        
        # Create ranking weights (exponential decay)
        weights = {}
        for rank, agent_id in enumerate(performance_ranking):
            if agent_id in agent_ids:
                # Higher rank = higher weight
                weights[agent_id] = np.exp(-rank * 0.3)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return RewardSharing.equal_sharing(rewards, agent_ids)
        
        shared_rewards = {}
        for agent_id in agent_ids:
            weight = weights.get(agent_id, 0.0)
            shared_rewards[agent_id] = (weight / total_weight) * total_reward
        
        return shared_rewards
    
    @staticmethod
    def team_bonus(
        rewards: Dict[str, float],
        agent_ids: List[str],
        team_performance: float,
        bonus_factor: float = 0.2
    ) -> Dict[str, float]:
        """
        Add team performance bonus to individual rewards
        
        Args:
            rewards: Dictionary of agent_id -> reward
            agent_ids: List of all agent IDs
            team_performance: Overall team performance metric
            bonus_factor: Factor for bonus calculation
            
        Returns:
            Dictionary of agent_id -> reward with bonus
        """
        bonus = team_performance * bonus_factor
        shared_rewards = rewards.copy()
        
        for agent_id in agent_ids:
            shared_rewards[agent_id] = shared_rewards.get(agent_id, 0.0) + bonus / len(agent_ids)
        
        return shared_rewards

