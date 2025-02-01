"""
Model utilities for language model fine-tuning.

This module provides utility functions for:
- Loading pre-trained models and tokenizers
- Setting up training arguments
- Handling model configuration

The utilities handle common setup tasks like:
- Padding token configuration
- Device management
- Training argument creation from config objects
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from typing import Tuple, Any

def load_model_and_tokenizer(model_name: str) -> Tuple[Any, AutoTokenizer]:
    """
    Load and configure a pre-trained model and its tokenizer.

    This function:
    1. Loads the model and tokenizer from Hugging Face hub
    2. Configures padding tokens if not present
    3. Ensures model and tokenizer are compatible

    Args:
        model_name (str): Name of the pre-trained model from Hugging Face hub
            Examples:
            - "distilgpt2" (smaller, faster)
            - "gpt2" (base model)
            - "gpt2-medium" (larger model)
            - "bert-base-uncased" (for classification tasks)

    Returns:
        Tuple[Any, AutoTokenizer]: Tuple containing:
            - model: The loaded and configured model
            - tokenizer: The associated tokenizer

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer("distilgpt2")
        >>> # Model and tokenizer are ready for training
    
    Note:
        If the model has no padding token, uses EOS token as padding
        This is common for GPT-2 style models
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def create_training_args(config) -> TrainingArguments:
    """
    Create a TrainingArguments object from configuration.

    Converts a configuration object into Hugging Face TrainingArguments,
    setting up all necessary training parameters.

    Args:
        config: Configuration object containing training parameters
            Required attributes:
            - output_dir: Directory for saving model
            - num_train_epochs: Number of training epochs
            - per_device_train_batch_size: Training batch size
            - per_device_eval_batch_size: Evaluation batch size
            - warmup_steps: Learning rate warmup steps
            - weight_decay: L2 regularization factor
            - logging_dir: Directory for logs
            - logging_steps: Frequency of logging
            - save_steps: Frequency of saving
            - eval_steps: Frequency of evaluation

    Returns:
        TrainingArguments: Configured training arguments object

    Example:
        >>> from config import TrainingConfig
        >>> config = TrainingConfig()
        >>> training_args = create_training_args(config)
        >>> # TrainingArguments ready for trainer initialization
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
    )
