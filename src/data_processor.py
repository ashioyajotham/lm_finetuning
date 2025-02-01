"""
Dataset processing module for language model fine-tuning.

This module handles:
- Loading datasets from Hugging Face hub
- Tokenization of text data
- Dataset preparation and preprocessing
- Batch processing of large datasets
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Any

class DataProcessor:
    """
    Handles dataset loading and preprocessing for language model training.
    
    This class provides functionality for:
    - Loading datasets from Hugging Face hub
    - Applying tokenization to text data
    - Preparing datasets for training
    - Handling batch processing
    
    Attributes:
        tokenizer (AutoTokenizer): Tokenizer matching the model architecture
        max_length (int): Maximum sequence length for tokenization
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> processor = DataProcessor(tokenizer, max_length=128)
        >>> dataset = processor.prepare_dataset("wikitext", "wikitext-2-raw-v1")
    """

    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        """
        Initialize the DataProcessor.

        Args:
            tokenizer (AutoTokenizer): Pre-trained tokenizer for text processing
            max_length (int): Maximum sequence length for tokenization
                Typical values:
                - 128: For testing or small models
                - 512: Standard for most tasks
                - 1024: For tasks requiring longer context
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self, dataset_name: str, dataset_config_name: str = None):
        """
        Load a dataset from Hugging Face hub.

        Args:
            dataset_name (str): Name of the dataset on Hugging Face hub
                Examples:
                - "wikitext"
                - "bookcorpus"
                - "squad"
            dataset_config_name (str, optional): Specific configuration name
                Examples for wikitext:
                - "wikitext-2-raw-v1"
                - "wikitext-103-raw-v1"

        Returns:
            datasets.Dataset: Loaded dataset object

        Raises:
            ValueError: If dataset cannot be found or loaded
        """
        return load_dataset(dataset_name, dataset_config_name)

    def tokenize_function(self, examples: Dict[str, Any]):
        """
        Tokenize a batch of text examples.

        Performs:
        1. Text tokenization
        2. Padding to max_length
        3. Truncation of longer sequences

        Args:
            examples (Dict[str, Any]): Batch of examples to tokenize
                Must contain 'text' key with string values

        Returns:
            Dict: Tokenized examples with:
                - input_ids
                - attention_mask
                - (optional) token_type_ids

        Note:
            Uses self.max_length for sequence length
            Applies padding='max_length' for consistent tensor shapes
        """
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

    def prepare_dataset(self, dataset_name: str, dataset_config_name: str = None):
        """
        Load and prepare dataset for training.

        Complete pipeline that:
        1. Loads dataset from Hugging Face
        2. Applies tokenization to all examples
        3. Removes unnecessary columns
        4. Prepares for training

        Args:
            dataset_name (str): Name of dataset to load
            dataset_config_name (str, optional): Specific configuration

        Returns:
            datasets.Dataset: Processed dataset ready for training
                Contains:
                - input_ids
                - attention_mask
                - (optional) token_type_ids

        Example:
            >>> processor = DataProcessor(tokenizer, max_length=128)
            >>> dataset = processor.prepare_dataset("wikitext", "wikitext-2-raw-v1")
            >>> # Dataset is ready for training
        """
        dataset = self.load_dataset(dataset_name, dataset_config_name)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        return tokenized_dataset
