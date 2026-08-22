"""
Solutions module storing best-known tour lengths and gap metrics calculation.
"""
from config import BEST_KNOWN_SOLUTIONS

def get_best_known(instance_name: str) -> float:
    clean_name = instance_name.split('.')[0]
    if clean_name not in BEST_KNOWN_SOLUTIONS:
        raise ValueError(f"Unknown instance name '{instance_name}'. Known instances: {list(BEST_KNOWN_SOLUTIONS.keys())}")
    return float(BEST_KNOWN_SOLUTIONS[clean_name])

def get_gap(current_length: float, instance_name: str) -> float:
    best = get_best_known(instance_name)
    if best <= 0:
        return 0.0
    return ((current_length - best) / best) * 100.0
