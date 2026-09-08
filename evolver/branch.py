from typing import Dict, List
import uuid
import random

class Branch:
    def __init__(self, id: str, signature: str):
        self.id = id
        self.signature = signature

    def generate_strategy(self, context: str = None, criticality_score: float = 0.9, randomness_factor: float = 0.2) -> Dict[str, any]:
        self._validate_inputs(context, criticality_score, randomness_factor)
        diversity = self._calculate_diversity()
        resource_usage = self._assess_resource_consumption(diversity)
        efficiency_score = self._calculate_efficiency(resource_usage)
        return self._build_strategy_response(diversity, resource_usage, efficiency_score, context)

    def _validate_inputs(self, context: str, criticality_score: float, randomness_factor: float) -> None:
        if not isinstance(context, str) and context is not None:
            raise ValueError('context must be a string or None')
        if not (0 <= criticality_score <= 1):
            raise ValueError('criticality_score must be between 0 and 1')
        if not (0 <= randomness_factor <= 1):
            raise ValueError('randomness_factor must be between 0 and 1')

    def _calculate_diversity(self) -> float:
        return round(self._calculate_random_value(0.2, 1.0), 2)

    def _assess_resource_consumption(self, diversity: float) -> float:
        return round(self._calculate_random_value(0.2, 1.0), 2)

    def _calculate_efficiency(self, resource_usage: float) -> float:
        return round(1.0 - (resource_usage * 0.4), 2)

    def _calculate_random_value(self, min_val: float, max_val: float) -> float:
        return round(random.uniform(min_val, max_val), 2)

    def _build_strategy_response(self, diversity: float, resource_usage: float, efficiency_score: float, context: str) -> Dict[str, any]:
        return {
            'diversity': diversity,
            'resource_usage': resource_usage,
            'efficiency_score': efficiency_score,
            'context': context
        }