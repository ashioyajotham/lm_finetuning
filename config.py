"""
Configuration module for language model fine-tuning.

This module contains dataclasses that define the configuration parameters
for model training and dataset processing.
"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Configuration for training parameters and model settings.
    
    Attributes:
        model_name (str): Name of the pre-trained model from Hugging Face.
            Options: "distilgpt2" (faster), "gpt2" (better quality),
            "gpt2-medium" (larger), "bert-base-uncased" (for classification)
        
        output_dir (str): Directory where model checkpoints and results will be saved
        
        num_train_epochs (int): Number of complete passes through the dataset
            Lower values (1-3) for testing, higher (3-10) for better results
        
        per_device_train_batch_size (int): Batch size for training
            Reduce this if you encounter CUDA out of memory errors
        
        per_device_eval_batch_size (int): Batch size for evaluation
            Can typically be 2x the training batch size
        
        warmup_steps (int): Number of steps for learning rate warmup
            Usually 5-10% of total training steps
        
        weight_decay (float): L2 regularization factor
            Higher values (0.1-0.01) for smaller models
            Lower values (0.01-0.001) for larger models
        
        logging_dir (str): Directory for tensorboard logs
        
        logging_steps (int): Number of steps between logging updates
            Lower values (10-50) give more frequent updates
        
        save_steps (int): Number of steps between model checkpoint saves
        
        eval_steps (int): Number of steps between evaluations
    """
    model_name: str = "distilgpt2"
    output_dir: str = "./results"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_dir: str = "./logs"
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000

@dataclass
class DataConfig:
    """
    Configuration for dataset processing and preparation.
    
    Attributes:
        dataset_name (str): Name of the dataset from Hugging Face hub
            Common options:
            - "wikitext": Wikipedia articles (good for general language modeling)
            - "bookcorpus": Books content (good for creative text)
            - "squad": Question-answering dataset
            - "glue": General Language Understanding Evaluation benchmark
        
        dataset_config_name (str): Specific configuration/subset of the dataset
            For wikitext options are:
            - "wikitext-2-raw-v1": Smaller dataset (~2M words)
            - "wikitext-103-raw-v1": Larger dataset (~100M words)
        
        max_length (int): Maximum sequence length for tokenization
            - 128: Good for testing
            - 512: Standard for most tasks
            - 1024: For tasks requiring longer context
        
        train_test_split (float): Fraction of data to use for testing
            Typical values: 0.1 (10%) or 0.2 (20%)
    """
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    max_length: int = 128
    train_test_split: float = 0.1

@dataclass
class VisualizationConfig:
    """
    Configuration for training visualization and logging.
    
    Attributes:
        use_wandb (bool): Whether to use Weights & Biases
        wandb_project (str): Project name for W&B
        wandb_entity (str): Entity name for W&B
        use_tensorboard (bool): Whether to use TensorBoard
        viz_port (int): Port for TensorBoard server
    """
    use_wandb: bool = True
    wandb_project: str = "lm-finetuning"
    wandb_entity: str = "ashioyajotham"
    use_tensorboard: bool = True
    viz_port: int = 6006
