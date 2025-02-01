"""
Evaluation metrics for language model assessment.

This module provides:
- Mathematical reasoning metrics
- Logical reasoning metrics
- Chain-of-thought metrics
- Text similarity metrics
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
import re
from difflib import SequenceMatcher

def calculate_numerical_accuracy(predicted: float, target: float, tolerance: float = 0.01) -> float:
    """
    Calculate accuracy for numerical answers with tolerance.
    
    Args:
        predicted: Predicted numerical value
        target: Target numerical value
        tolerance: Acceptable difference (default: 0.01)
    """
    return float(abs(predicted - target) <= tolerance)

def evaluate_step_accuracy(generated_steps: List[str], target_steps: List[str]) -> float:
    """
    Evaluate accuracy of solution steps using sequence matching.
    
    Args:
        generated_steps: List of generated solution steps
        target_steps: List of expected solution steps
    
    Returns:
        float: Score between 0 and 1 indicating step accuracy
    """
    if not generated_steps or not target_steps:
        return 0.0
    
    total_score = 0.0
    for target in target_steps:
        step_scores = [
            SequenceMatcher(None, target.lower(), gen.lower()).ratio()
            for gen in generated_steps
        ]
        total_score += max(step_scores) if step_scores else 0
    
    return total_score / len(target_steps)

def extract_math_expressions(text: str) -> List[str]:
    """
    Extract mathematical expressions from text.
    
    Args:
        text: Input text containing mathematical expressions
    
    Returns:
        List of extracted mathematical expressions
    """
    # Match basic arithmetic expressions
    expression_pattern = r'\b\d+(?:\s*[-+*/]\s*\d+)+\b|\d+\s*=\s*\d+'
    return re.findall(expression_pattern, text)

def evaluate_reasoning_structure(text: str) -> Dict[str, float]:
    """
    Evaluate the structure of reasoning in text.
    
    Returns dict with scores for:
    - step_clarity: Clear step-by-step reasoning
    - logical_flow: Logical progression
    - completeness: Solution completeness
    """
    # Split into steps
    steps = [s.strip() for s in text.split('\n') if s.strip()]
    
    metrics = {
        "step_clarity": 0.0,
        "logical_flow": 0.0,
        "completeness": 0.0
    }
    
    # Step clarity
    step_indicators = ['first', 'second', 'then', 'next', 'finally', 'step']
    metrics["step_clarity"] = sum(
        any(ind in step.lower() for ind in step_indicators)
        for step in steps
    ) / max(len(steps), 1)
    
    # Logical flow
    logical_connectors = ['therefore', 'because', 'thus', 'hence', 'so']
    metrics["logical_flow"] = sum(
        any(conn in step.lower() for conn in logical_connectors)
        for step in steps
    ) / max(len(steps), 1)
    
    # Completeness
    has_numbers = any(bool(re.search(r'\d', step)) for step in steps)
    has_conclusion = any(word in text.lower() for word in ['therefore', 'conclusion', 'answer'])
    has_multiple_steps = len(steps) > 1
    
    metrics["completeness"] = sum([
        has_numbers,
        has_conclusion,
        has_multiple_steps
    ]) / 3.0
    
    return metrics

def evaluate_mathematical_correctness(text: str) -> Dict[str, Union[float, List[str]]]:
    """
    Evaluate mathematical correctness of solution.
    
    Returns:
        Dict containing:
        - expression_validity: Score for valid mathematical expressions
        - arithmetic_accuracy: Score for correct calculations
        - invalid_expressions: List of potentially incorrect expressions
    """
    expressions = extract_math_expressions(text)
    results = {
        "expression_validity": 0.0,
        "arithmetic_accuracy": 0.0,
        "invalid_expressions": []
    }
    
    if not expressions:
        return results
    
    valid_count = 0
    correct_count = 0
    
    for expr in expressions:
        try:
            # Check if it's an equation
            if '=' in expr:
                left, right = expr.split('=')
                left_val = eval(left.strip())
                right_val = eval(right.strip())
                is_valid = abs(left_val - right_val) < 0.01
            else:
                # Evaluate expression
                eval(expr)
                is_valid = True
            
            valid_count += 1
            correct_count += int(is_valid)
            
            if not is_valid:
                results["invalid_expressions"].append(expr)
                
        except Exception:
            results["invalid_expressions"].append(expr)
            continue
    
    total = len(expressions)
    results["expression_validity"] = valid_count / total if total > 0 else 0
    results["arithmetic_accuracy"] = correct_count / total if total > 0 else 0
    
    return results

def aggregate_math_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate multiple evaluation metrics.
    
    Args:
        metrics_list: List of metric dictionaries to aggregate
    
    Returns:
        Dictionary of averaged metrics
    """
    if not metrics_list:
        return {}
        
    aggregated = {}
    for metric_dict in metrics_list:
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)):
                aggregated[key] = aggregated.get(key, 0.0) + value
    
    return {k: v / len(metrics_list) for k, v in aggregated.items()}
