"""
Evaluation utilities for language model assessment.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datasets import load_dataset, load_metric
import json
import math
import re
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from datetime import datetime
import pandas as pd
from scipy import stats

class NLGMetrics:
    """Standard NLG metrics implementation"""
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.meteor = load_metric('meteor')
        self.smooth = SmoothingFunction().method1

    def compute_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        return {
            'bleu': sentence_bleu([reference.split()], generated.split(), smoothing_function=self.smooth),
            **self.rouge_scorer.score(reference, generated),
            'meteor': self.meteor.compute(predictions=[generated], references=[reference])['meteor']
        }

class ABTesting:
    """A/B testing framework for model comparison"""
    def __init__(self, model_a, model_b, tokenizer):
        self.model_a = model_a
        self.model_b = model_b
        self.tokenizer = tokenizer
        self.results = []

    def run_test(self, test_cases: List[Dict[str, str]], metrics: List[Callable]) -> Dict[str, Any]:
        results_a = self._evaluate_model(self.model_a, test_cases, metrics)
        results_b = self._evaluate_model(self.model_b, test_cases, metrics)
        
        statistical_tests = self._run_statistical_tests(results_a, results_b)
        return {
            'model_a_results': results_a,
            'model_b_results': results_b,
            'statistical_tests': statistical_tests
        }

    def _run_statistical_tests(self, results_a: List[float], results_b: List[float]) -> Dict[str, float]:
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        return {'t_statistic': t_stat, 'p_value': p_value}

class HumanEvaluation:
    """Human-in-the-loop evaluation framework"""
    def __init__(self, save_path: str = "human_evaluations.json"):
        self.save_path = save_path
        self.evaluations = []

    def add_evaluation(self, generated_text: str, human_score: int, feedback: str, 
                      criteria: Dict[str, int]) -> None:
        """Add human evaluation with structured feedback"""
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'generated_text': generated_text,
            'overall_score': human_score,
            'feedback': feedback,
            'criteria_scores': criteria
        }
        self.evaluations.append(evaluation)
        self._save_evaluations()

    def get_aggregate_scores(self) -> Dict[str, float]:
        """Get aggregate scores across all evaluations"""
        if not self.evaluations:
            return {}
            
        df = pd.DataFrame(self.evaluations)
        return {
            'mean_score': df['overall_score'].mean(),
            'criteria_means': {
                criterion: [e['criteria_scores'][criterion] for e in self.evaluations if criterion in e['criteria_scores']]
                for criterion in self.evaluations[0]['criteria_scores'].keys()
            }
        }

class ModelEvaluator:
    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.numerical_tolerance = 0.01  # For floating point comparisons
        self.nlg_metrics = NLGMetrics()
        self.human_eval = HumanEvaluation()

    def calculate_perplexity(self, text: str, stride: int = 512) -> float:
        """Calculate perplexity for a given text."""
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.max_position_embeddings
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        return torch.exp(torch.stack(nlls).mean()).item()

    def evaluate_mathematical_reasoning(self, examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate mathematical reasoning capabilities"""
        total_correct = 0
        total_step_accuracy = 0
        category_scores = {}
        
        for example in examples:
            # Generate solution for the question
            prompt = f"Question: {example['question']}\nSolution:"
            generated = self.generate_solution(prompt)
            
            # Check numerical answer accuracy
            answer_correct = self.check_numerical_answer(generated, example['answer'])
            
            # Evaluate solution steps
            step_acc = self.evaluate_solution_steps(generated, example['steps'])
            
            # Update scores
            total_correct += answer_correct
            total_step_accuracy += step_acc
            
            # Track category performance
            category = example['category']
            if category not in category_scores:
                category_scores[category] = {"correct": 0, "total": 0}
            category_scores[category]["correct"] += answer_correct
            category_scores[category]["total"] += 1

        # Calculate final metrics
        results = {
            "overall_accuracy": total_correct / len(examples),
            "step_accuracy": total_step_accuracy / len(examples),
        }
        
        # Add category-specific accuracies
        for category, scores in category_scores.items():
            results[f"{category}_accuracy"] = scores["correct"] / scores["total"]
            
        return results

    def evaluate_logical_reasoning(self, examples: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate logical reasoning capabilities."""
        metrics = {
            "premise_usage": 0,
            "conclusion_validity": 0,
            "logical_consistency": 0
        }
        
        for example in examples:
            prompt = (f"Premises: {example['premises']}\n"
                     f"Question: {example['question']}\n"
                     "Provide a logical step-by-step solution:")
            
            generated = self.generate_solution(prompt)
            
            # Evaluate premise usage
            metrics["premise_usage"] += self.check_premise_usage(
                generated, example['premises']
            )
            
            # Evaluate conclusion
            if 'conclusion' in example:
                metrics["conclusion_validity"] += self.check_conclusion(
                    generated, example['conclusion']
                )
            
            # Evaluate logical consistency
            metrics["logical_consistency"] += self.check_logical_consistency(generated)

        # Average the metrics
        return {k: v/len(examples) for k, v in metrics.items()}

    def evaluate_chain_of_thought(self, examples: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate chain-of-thought reasoning."""
        metrics = {
            "step_completion": 0,
            "reasoning_coherence": 0,
            "conclusion_alignment": 0
        }
        
        for example in examples:
            prompt = f"{example['prompt']}\nThink about this step by step:"
            generated = self.generate_solution(prompt)
            
            # Evaluate step completion
            metrics["step_completion"] += self.evaluate_steps(
                generated, example.get('expected_steps', [])
            )
            
            # Evaluate reasoning coherence
            metrics["reasoning_coherence"] += self.evaluate_coherence(generated)
            
            # Evaluate conclusion alignment
            if 'conclusion' in example:
                metrics["conclusion_alignment"] += self.check_conclusion_alignment(
                    generated, example['conclusion']
                )

        return {k: v/len(examples) for k, v in metrics.items()}

    def generate_solution(self, prompt: str) -> str:
        """Generate solution for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            num_return_sequences=1,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def check_numerical_answer(self, generated: str, correct_answer: str) -> bool:
        """Check if generated answer matches the correct answer within tolerance"""
        try:
            # Extract numerical value from generated text
            numbers = re.findall(r"[-+]?\d*\.?\d+", generated)
            if not numbers:
                return False
            generated_num = float(numbers[-1])  # Take last number as answer
            correct_num = float(correct_answer)
            return abs(generated_num - correct_num) <= self.numerical_tolerance
        except ValueError:
            return False

    def evaluate_solution_steps(self, generated: str, correct_steps: List[str]) -> float:
        """Evaluate solution steps for completeness and correctness"""
        # Split generated text into steps
        generated_steps = [s.strip() for s in generated.split('\n') if s.strip()]
        
        # Calculate step coverage
        step_scores = []
        for correct_step in correct_steps:
            step_score = max(
                SequenceMatcher(None, correct_step.lower(), gen_step.lower()).ratio()
                for gen_step in generated_steps
            )
            step_scores.append(step_score)
        
        return np.mean(step_scores) if step_scores else 0.0

    # Helper methods for evaluation
    def extract_numerical_answer(self, text: str) -> float:
        """Extract numerical answer from generated text."""
        try:
            # Find the last number in the text
            words = text.split()
            for word in reversed(words):
                try:
                    return float(word.strip('.,()'))
                except ValueError:
                    continue
            return 0.0
        except Exception:
            return 0.0

    def check_premise_usage(self, generated: str, premises: str) -> float:
        """Check if premises are used in the reasoning."""
        premise_keywords = set(premises.lower().split())
        generated_words = set(generated.lower().split())
        overlap = len(premise_keywords.intersection(generated_words))
        return overlap / len(premise_keywords) if premise_keywords else 0

    def check_logical_consistency(self, text: str) -> float:
        """Check for logical consistency in the reasoning."""
        # Simple implementation - look for logical connectors
        logical_connectors = ['therefore', 'because', 'thus', 'hence', 'so']
        text_lower = text.lower()
        connector_count = sum(text_lower.count(c) for c in logical_connectors)
        return min(1.0, connector_count / 3)  # Normalize to [0,1]

    def evaluate_coherence(self, text: str) -> float:
        """Evaluate the coherence of the reasoning."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.0
            
        # Check for step indicators
        step_indicators = ['first', 'second', 'then', 'next', 'finally']
        indicator_count = sum(
            any(ind in sent.lower() for ind in step_indicators)
            for sent in sentences
        )
        return min(1.0, indicator_count / len(sentences))

    def structured_evaluation(self, 
                            test_cases: List[Dict[str, Any]], 
                            workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a structured evaluation workflow"""
        results = {
            'automatic_metrics': {},
            'human_evaluation': {},
            'task_specific': {},
            'timestamps': {
                'start': datetime.now().isoformat(),
                'end': None
            }
        }
        
        # Automatic metrics
        if workflow_config.get('run_automatic_metrics', True):
            results['automatic_metrics'] = self._run_automatic_metrics(test_cases)
            
        # Task-specific metrics
        if workflow_config.get('task_specific_metrics'):
            results['task_specific'] = self._run_task_specific_metrics(
                test_cases, 
                workflow_config['task_specific_metrics']
            )
            
        # Human evaluation integration
        if workflow_config.get('human_evaluation'):
            results['human_evaluation'] = self._run_human_evaluation(
                test_cases,
                workflow_config['human_evaluation']
            )
            
        results['timestamps']['end'] = datetime.now().isoformat()
        return results

    def evaluate_task_specific(self, examples: List[Dict[str, Any]], task_type: str) -> Dict[str, float]:
        """Task-specific evaluation metrics"""
        if task_type == "math":
            return self.evaluate_mathematical_reasoning(examples)
        elif task_type == "logic":
            return self.evaluate_logical_reasoning(examples)
        elif task_type == "chain_of_thought":
            return self.evaluate_chain_of_thought(examples)
        elif task_type == "summarization":
            return self._evaluate_summarization(examples)
        elif task_type == "qa":
            return self._evaluate_qa(examples)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _evaluate_summarization(self, examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluation metrics specific to summarization"""
        metrics = {
            'content_coverage': 0.0,
            'conciseness': 0.0,
            'coherence': 0.0
        }
        
        for example in examples:
            generated = self.generate_solution(example['text'])
            reference = example['summary']
            
            # Content coverage using ROUGE
            rouge_scores = self.nlg_metrics.rouge_scorer.score(reference, generated)
            metrics['content_coverage'] += rouge_scores['rougeL'].fmeasure
            
            # Conciseness score
            metrics['conciseness'] += min(1.0, len(reference.split()) / len(generated.split()))
            
            # Coherence using sentence similarity
            metrics['coherence'] += self.evaluate_coherence(generated)
            
        return {k: v/len(examples) for k, v in metrics.items()}

    def _evaluate_qa(self, examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluation metrics specific to question-answering"""
        metrics = {
            'exact_match': 0.0,
            'f1_score': 0.0,
            'answer_relevance': 0.0
        }
        
        for example in examples:
            context = example.get('context', '')
            question = example['question']
            generated = self.generate_solution(f"Context: {context}\nQuestion: {question}")
            
            # Exact match
            metrics['exact_match'] += int(generated.lower() == example['answer'].lower())
            
            # F1 score for partial matches
            metrics['f1_score'] += self._calculate_f1(generated, example['answer'])
            
            # Answer relevance using semantic similarity
            metrics['answer_relevance'] += self._calculate_relevance(generated, example['answer'])
            
        return {k: v/len(examples) for k, v in metrics.items()}
