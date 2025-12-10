"""
Data collection and quality assessment tools
"""

import numpy as np
from typing import Dict, List, Any
import random


class DataSource:
    """Simulated data source"""
    
    def __init__(
        self,
        source_id: str,
        source_type: str,
        quality: float = 0.7,
        availability: float = 0.9
    ):
        """
        Initialize data source
        
        Args:
            source_id: Unique identifier
            source_type: Type of source (api, database, web, etc.)
            quality: Data quality score (0-1)
            availability: Availability probability (0-1)
        """
        self.source_id = source_id
        self.source_type = source_type
        self.quality = quality
        self.availability = availability
        self.collection_count = 0
        
    def collect(self) -> Dict[str, Any]:
        """Collect data from source"""
        if random.random() > self.availability:
            return {'status': 'unavailable', 'source_id': self.source_id}
        
        self.collection_count += 1
        
        # Simulate data collection
        data = {
            'id': f"data_{self.source_id}_{self.collection_count}",
            'source_id': self.source_id,
            'source_type': self.source_type,
            'quality': self.quality + np.random.normal(0, 0.1),
            'size': random.randint(100, 1000),
            'timestamp': self.collection_count
        }
        
        return data
    
    def get_info(self) -> Dict[str, Any]:
        """Get source information"""
        return {
            'id': self.source_id,
            'type': self.source_type,
            'quality': self.quality,
            'availability': self.availability,
            'collection_count': self.collection_count
        }


class DataQualityAssessor:
    """Tool for assessing data quality"""
    
    @staticmethod
    def assess_quality(data: Dict[str, Any]) -> float:
        """
        Assess quality of collected data
        
        Args:
            data: Data dictionary
            
        Returns:
            Quality score (0-1)
        """
        quality = data.get('quality', 0.5)
        
        # Additional factors
        if 'size' in data:
            size_score = min(data['size'] / 1000.0, 1.0)
            quality = (quality + size_score) / 2
        
        return np.clip(quality, 0.0, 1.0)
    
    @staticmethod
    def filter_by_quality(data_list: List[Dict[str, Any]], min_quality: float = 0.6) -> List[Dict[str, Any]]:
        """
        Filter data by quality threshold
        
        Args:
            data_list: List of data dictionaries
            min_quality: Minimum quality threshold
            
        Returns:
            Filtered data list
        """
        return [
            data for data in data_list
            if DataQualityAssessor.assess_quality(data) >= min_quality
        ]

