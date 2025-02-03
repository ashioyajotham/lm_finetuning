"""
Main training and generation script for language model fine-tuning.

This script provides a unified interface for:
1. Training new models
2. Continuing training from checkpoints
3. Generating text from trained models

Key Features:
- Flexible model selection from HuggingFace hub
- Dataset configuration and processing
- Training visualization with TensorBoard/W&B
- Text generation with configurable parameters

Commands:
    Training:
        python train.py --mode train --model gpt2 --dataset wikitext
        
    Continue Training:
        python train.py --mode continue --continue_from ./checkpoint-1000
        
    Generate Text:
        python train.py --mode generate --model_path ./results/final_model \
                       --prompt "Once upon a time" \
                       --temperature 0.8 --max_length 200

Configuration:
    - Training settings in TrainingConfig
    - Dataset options in DataConfig
    - Visualization settings in VisualizationConfig
"""

import argparse
from pathlib import Path
import json
import os
from datetime import datetime

from src.data_processor import DataProcessor
from src.model_utils import load_model_and_tokenizer, create_training_args
from src.trainer import CustomTrainer
from config import TrainingConfig, DataConfig

def save_training_metadata(output_dir: str, config: dict):
    """
    Save training configuration and metadata for reproducibility.

    Args:
        output_dir (str): Directory to save metadata
        config (dict): Configuration to save, including:
            - training_config: Model and training parameters
            - data_config: Dataset settings
            - cli_args: Command line arguments used
            
    Creates:
        training_metadata.json with:
        - Training date and time
        - Complete configuration
        - CLI arguments used
    """
    metadata = {
        "training_date": datetime.now().isoformat(),
        "config": config,
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def main(args):
    """
    Main function handling model training, continuing training, and text generation.

    Modes:
        train: Train a new model
            Required: --model (optional, default: distilgpt2)
            Optional: --dataset, --wandb
            
        continue: Continue training from checkpoint
            Required: --continue_from (checkpoint path)
            Optional: --wandb
            
        generate: Generate text using a trained model
            Required: --model_path
            Optional: --prompt, --max_length, --temperature, --top_k, --top_p

    Args:
        args: Command line arguments including:
            mode (str): Operation mode (train/continue/generate)
            model (str): Model name from HuggingFace
            dataset (str): Dataset name from HuggingFace
            continue_from (str): Path to checkpoint
            model_path (str): Path to model for generation
            prompt (str): Text prompt for generation
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p sampling parameter
            wandb (bool): Enable Weights & Biases logging

    Examples:
        >>> # Train new model
        >>> args = parser.parse_args(['--mode', 'train', '--model', 'gpt2'])
        >>> main(args)
        
        >>> # Generate text
        >>> args = parser.parse_args(['--mode', 'generate', 
        ...                          '--model_path', './results/final_model',
        ...                          '--prompt', 'Once upon a time'])
        >>> main(args)
    """
    # Load configurations
    training_config = TrainingConfig()
    data_config = DataConfig()

    # Update configs from CLI args
    if args.model:
        training_config.model_name = args.model
    if args.dataset:
        data_config.dataset_name = args.dataset
        
    # Handle model loading
    if args.mode == "generate":
        if not args.model_path:
            raise ValueError("--model_path required for generation mode")
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        print(f"Loaded model from: {args.model_path}")
    else:  # training modes
        if args.continue_from:
            model, tokenizer = load_model_and_tokenizer(args.continue_from)
            print(f"Continuing training from: {args.continue_from}")
        else:
            model, tokenizer = load_model_and_tokenizer(training_config.model_name)

    # Prepare dataset
    data_processor = DataProcessor(tokenizer, data_config.max_length)
    dataset = data_processor.prepare_dataset(
        data_config.dataset_name,
        data_config.dataset_config_name
    )

    # Set up trainer
    trainer = CustomTrainer(
        model=model,
        args=create_training_args(training_config),
        train_dataset=dataset["train"] if args.mode != "generate" else None,
        eval_dataset=dataset["test"] if args.mode != "generate" and "test" in dataset else None,
        tokenizer=tokenizer,
        viz_config={'use_tensorboard': True, 'use_wandb': args.wandb}
    )

    # Handle different modes
    if args.mode == "evaluate":
        if not args.eval_file:
            raise ValueError("--eval_file required for evaluation mode")
            
        # Load workflow configuration
        with open(args.eval_workflow) as f:
            workflow_config = json.load(f)
        
        if args.human_eval_template:
            with open(args.human_eval_template) as f:
                human_eval_config = json.load(f)
                workflow_config['human_evaluation_template'] = human_eval_config
            
        # Load evaluation examples
        with open(args.eval_file) as f:
            eval_examples = json.load(f)
            
        # Run structured evaluation
        results = trainer.run_structured_evaluation(eval_examples, workflow_config)
        
        # Print detailed results in a structured way
        print("\nEvaluation Results:")
        for category in ['automatic_metrics', 'task_specific', 'examples']:
            if category in results:
                print(f"\n{category.upper()}:")
                if isinstance(results[category], dict):
                    for metric, value in results[category].items():
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
                elif isinstance(results[category], list):
                    print(f"  Number of examples: {len(results[category])}")
        
        # Print timestamps
        if 'timestamps' in results:
            print(f"\nEvaluation started: {results['timestamps']['start']}")
            print(f"Evaluation ended: {results['timestamps']['end']}")
            
    elif args.mode == "generate":
        # Add generation configuration
        gen_config = {
            "max_length": args.max_length or 150,
            "temperature": args.temperature or 0.9,
            "top_k": args.top_k or 50,
            "top_p": args.top_p or 0.95,
            "do_sample": True,  # Enable sampling
            "pad_token_id": tokenizer.eos_token_id,
            "num_return_sequences": 1
        }

        prompts = [args.prompt] if args.prompt else [
            "Once upon a time",
            "In a world where",
            "In the future",
            "Lock in",
            "Building and accelerating"
        ]
        
        for prompt in prompts:
            # Encode prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate
            outputs = model.generate(
                **inputs,
                **gen_config
            )
            
            # Decode and remove prompt from output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in generated_text:
                generated_text = generated_text[len(prompt):].strip()
            
            print(f"\nPrompt: {prompt}\nGenerated: {generated_text}\n{'-'*50}")
    else:
        # Training modes
        trainer.train(resume_from_checkpoint=args.continue_from)
        model.save_pretrained(f"{training_config.output_dir}/final_model")
        tokenizer.save_pretrained(f"{training_config.output_dir}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Language Model Fine-tuning and Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train a new model:
    python train.py --mode train --model gpt2 --dataset wikitext

    # Continue training from checkpoint:
    python train.py --mode continue --continue_from ./results/checkpoint-1000

    # Generate text:
    python train.py --mode generate --model_path ./results/final_model \
                   --prompt "Once upon a time" --temperature 0.8

    # Enable W&B logging:
    python train.py --mode train --model gpt2 --wandb
        """
    )
    
    parser.add_argument("--mode", choices=["train", "continue", "generate", "evaluate"], 
                       default="train",
                       help="Operation mode: train/continue/generate/evaluate")
    
    # Training arguments
    parser.add_argument("--model", type=str,
                       help="Model name from HuggingFace hub")
    parser.add_argument("--dataset", type=str,
                       help="Dataset name from HuggingFace hub")
    parser.add_argument("--continue_from", type=str,
                       help="Path to checkpoint to continue training from")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    
    # Generation arguments
    parser.add_argument("--model_path", type=str,
                       help="Path to trained model for generation")
    parser.add_argument("--prompt", type=str,
                       help="Text prompt for generation")
    parser.add_argument("--max_length", type=int,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float,
                       help="Top-p sampling parameter")
    
    # Evaluation arguments
    parser.add_argument("--eval_file", type=str,
                      help="JSON file containing evaluation examples")
    parser.add_argument("--eval_workflow", type=str,
                      help="Path to evaluation workflow configuration",
                      default="evaluation_workflows/default_workflow.json")
    parser.add_argument("--human_eval_template", type=str,
                      help="Path to human evaluation template file",
                      default="evaluation_workflows/human_eval_template.json")
    
    args = parser.parse_args()
    main(args)
