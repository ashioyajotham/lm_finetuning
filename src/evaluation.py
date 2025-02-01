"""
Evaluation utilities for language model assessment.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from datasets import load_dataset
import json
import math

class ModelEvaluator:
    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

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

    def evaluate_mathematical_reasoning(self, examples: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate mathematical reasoning capabilities."""
        correct = 0
        total = len(examples)
        step_accuracy = []

        for example in examples:
            prompt = f"Problem: {example['question']}\nSolve this step by step:"
            generated = self.generate_solution(prompt)
            
            # Evaluate numerical answer
            if 'answer' in example:
                pred_answer = self.extract_numerical_answer(generated)
                correct += int(abs(pred_answer - float(example['answer'])) < 0.01)
            
            # Evaluate solution steps
            if 'steps' in example:
                step_acc = self.evaluate_solution_steps(generated, example['steps'])
                step_accuracy.append(step_acc)

        return {
            "accuracy": correct / total if total > 0 else 0,
            "step_accuracy": np.mean(step_accuracy) if step_accuracy else 0
        }

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
