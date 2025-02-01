"""
Custom Trainer module for fine-tuning language models.

This module extends the Hugging Face Trainer class to provide:
- Custom loss computation
- Enhanced evaluation metrics including perplexity
- Text generation capabilities
- Specialized prediction steps
"""

from transformers import Trainer
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset
import numpy as np
from transformers.trainer_utils import EvalPrediction

# Custom imports
from .visualization import TrainingVisualizer
from .evaluation import ModelEvaluator

class CustomTrainer(Trainer):
    """
    CustomTrainer extends the Hugging Face Trainer for language model fine-tuning.
    
    This trainer adds specialized functionality for:
    - Handling language modeling loss computation
    - Computing perplexity during evaluation
    - Generating text from prompts
    - Supporting both labeled and unlabeled datasets
    
    The trainer automatically handles:
    - Device management (CPU/GPU)
    - Gradient computation and optimization
    - Model checkpointing
    - Training logs
    """
    def __init__(self, *args, viz_config=None, **kwargs):
        # Handle deprecated tokenizer warning by using processing_class
        if 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
            
        super().__init__(*args, **kwargs)
        self.processor = kwargs.get('processing_class', None)
        
        # Initialize visualizer with proper config and model
        self.visualizer = TrainingVisualizer(
            self.args,
            self.model.config._name_or_path,
            model=self.model,  # Pass the model instance
            viz_config=viz_config
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss for language modeling.
        
        This method handles both labeled and unlabeled data by:
        1. Using provided labels if available
        2. Creating shifted labels from input_ids if no labels provided
        3. Scaling loss based on batch size when needed
        
        Args:
            model: The language model being trained
            inputs (dict): Input tensors including input_ids and optional labels
            return_outputs (bool): Whether to return model outputs along with loss
            num_items_in_batch (Optional[int]): For loss scaling in gradient accumulation
        
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: Loss or (loss, outputs)
        """
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels") if "labels" in inputs else inputs["input_ids"].clone()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        if num_items_in_batch is not None:
            loss = loss * num_items_in_batch / len(inputs["input_ids"])

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        **kwargs
    ) -> EvalPrediction:
        """
        Custom evaluation loop with perplexity calculation.
        
        Performs model evaluation by:
        1. Computing loss on evaluation dataset
        2. Calculating perplexity (exp(loss)) as additional metric
        3. Aggregating metrics across all batches
        
        Args:
            dataloader: DataLoader for evaluation data
            description: Description string for progress bar
            prediction_loss_only: Whether to only compute loss
            **kwargs: Additional arguments passed to parent class
        
        Returns:
            EvalPrediction: Object containing evaluation metrics
                - eval_loss: Average loss across batches
                - perplexity: exp(eval_loss)
        """
        eval_losses = []
        model = self.model.eval()
        
        for batch in dataloader:
            with torch.no_grad():
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                loss, outputs = self.compute_loss(model, batch, return_outputs=True)
                eval_losses.append(loss.item())
        
        metrics = {
            "eval_loss": np.mean(eval_losses),
            "perplexity": np.exp(np.mean(eval_losses))
        }
        
        return metrics

    def prediction_step(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step for inference.
        
        Handles prediction by:
        1. Managing input/label separation
        2. Computing loss if needed
        3. Generating logits for token prediction
        
        Args:
            model: The language model
            inputs: Dictionary of input tensors
            prediction_loss_only: Whether to only return loss
        
        Returns:
            Tuple containing:
            - loss: Optional[float] - The loss value if computed
            - logits: Optional[torch.Tensor] - Prediction logits
            - labels: Optional[torch.Tensor] - Ground truth labels
        """
        with torch.no_grad():
            if "labels" in inputs:
                labels = inputs["labels"]
            else:
                labels = inputs["input_ids"].clone()
            
            outputs = model(**inputs)
            loss = outputs.loss

            logits = outputs.logits
            
            return (loss.item(), logits, labels)

    def generate_text(
        self, 
        prompt: str, 
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = True,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Generate higher quality text continuation from a prompt.
        
        Args:
            prompt (str): Input text to continue from
            max_length (int): Maximum length of generated sequence
            num_return_sequences (int): Number of sequences to generate
            temperature (float): Sampling temperature (higher = more creative)
            top_k (int): Number of highest probability tokens to keep
            top_p (float): Cumulative probability for nucleus sampling
            no_repeat_ngram_size (int): Size of n-grams to prevent repetition
            do_sample (bool): Whether to use sampling vs greedy generation
            repetition_penalty (float): Penalty for repeating tokens
        
        Returns:
            str: Generated text continuation
        """
        if self.processor is None:
            raise ValueError("No processor (tokenizer) available for text generation")

        inputs = self.processor(prompt, return_tensors="pt").to(self.args.device)
        
        # Set pad token if not set
        if self.processor.pad_token_id is None:
            self.processor.pad_token_id = self.processor.eos_token_id
            
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.processor.pad_token_id,
            bos_token_id=self.processor.bos_token_id if self.processor.bos_token_id else None,
            eos_token_id=self.processor.eos_token_id if self.processor.eos_token_id else None,
        )
        
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        if hasattr(self, 'visualizer'):
            self.visualizer.log_text_samples(
                prompt, 
                generated_text, 
                self.state.global_step
            )
        
        return generated_text

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log method to add visualization.
        
        Args:
            logs: Dict of log values
            start_time: Optional start time for calculating throughput
        """
        # Call parent class log method
        super().log(logs, start_time)
        
        # Add visualization if available
        if hasattr(self, 'visualizer'):
            # Filter out None values and convert tensors to floats
            clean_logs = {
                k: float(v) if hasattr(v, 'item') else v
                for k, v in logs.items()
                if v is not None
            }
            self.visualizer.log_metrics(clean_logs, self.state.global_step)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """Add additional metrics visualization"""
        output = super()._maybe_log_save_evaluate(*args, **kwargs)
        if output and hasattr(self, 'visualizer'):
            self.visualizer.log_metrics({
                'learning_rate': self.lr_scheduler.get_last_lr()[0],
                'memory_used': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            }, self.state.global_step)
        return output

    def run_evaluations(self, eval_examples: Dict[str, List[Dict[str, str]]]) -> Dict[str, float]:
        """
        Run comprehensive model evaluations.
        
        Args:
            eval_examples: Dictionary containing evaluation examples for different tasks
        """
        evaluator = ModelEvaluator(self.model, self.processor, self.args.device)
        
        results = {}
        
        # Run different types of evaluations
        if 'math' in eval_examples:
            math_results = evaluator.evaluate_mathematical_reasoning(eval_examples['math'])
            results.update({f"math_{k}": v for k, v in math_results.items()})
            
        if 'logic' in eval_examples:
            logic_results = evaluator.evaluate_logical_reasoning(eval_examples['logic'])
            results.update({f"logic_{k}": v for k, v in logic_results.items()})
            
        if 'cot' in eval_examples:
            cot_results = evaluator.evaluate_chain_of_thought(eval_examples['cot'])
            results.update({f"cot_{k}": v for k, v in cot_results.items()})
        
        # Log evaluation results
        if hasattr(self, 'visualizer'):
            self.visualizer.log_metrics(results, self.state.global_step)
        
        return results

    def __del__(self):
        """Cleanup visualizer on deletion"""
        if hasattr(self, 'visualizer'):
            self.visualizer.close()
