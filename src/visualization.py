"""
Visualization utilities for training monitoring.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import subprocess
from pathlib import Path
import threading
import webbrowser
import time
import numpy as np
from typing import Dict, Any, Optional
import wandb
import os

class TrainingVisualizer:
    def __init__(self, training_args, model_name: str, model=None, viz_config=None):
        """Initialize visualization tools based on config"""
        self.training_args = training_args
        self.model_name = model_name
        # Store the model that's passed in
        self.model = model
        self.writer = None
        self.wandb_run = None
        
        # Set comprehensive default config
        default_config = {
            'use_tensorboard': True,
            'use_wandb': False,
            'wandb_project': 'lm-finetuning',
            'viz_port': 6006  # Add default port
        }
        
        # Update defaults with provided config if any
        self.viz_config = default_config
        if viz_config:
            self.viz_config.update(viz_config)
        
        if self.viz_config['use_tensorboard']:
            self.writer = SummaryWriter(training_args.logging_dir)
            self._launch_tensorboard()
            
        if self.viz_config['use_wandb']:
            # Initialize wandb with more detailed config
            self.wandb_run = wandb.init(
                project=self.viz_config['wandb_project'],
                name=f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model_name": model_name,
                    "training_args": vars(training_args),
                    "viz_config": self.viz_config,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                },
                notes="Language model fine-tuning run",
                tags=["fine-tuning", model_name],
            )
            
            # Only watch model if it's provided
            if self.model is not None:
                wandb.watch(
                    models=self.model,
                    log="all",
                    log_freq=100
                )

    def _launch_tensorboard(self):
        """Launch TensorBoard server in background"""
        def run_tensorboard():
            subprocess.run([
                "tensorboard",
                "--logdir", self.training_args.logging_dir,
                "--port", str(self.viz_config['viz_port'])
            ])

        thread = threading.Thread(target=run_tensorboard, daemon=True)
        thread.start()
        
        # Wait a moment for server to start
        time.sleep(3)
        webbrowser.open(f"http://localhost:{self.viz_config['viz_port']}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all active visualization platforms"""
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
                
        if self.wandb_run:
            # Add custom formatting for wandb
            wandb_metrics = {
                k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                for k, v in metrics.items()
            }
            self.wandb_run.log(wandb_metrics, step=step)

    def log_model_graph(self, model: torch.nn.Module, sample_input: torch.Tensor):
        """Log model architecture graph"""
        if self.writer:
            self.writer.add_graph(model, sample_input)

    def log_text_samples(self, prompt: str, generated: str, step: int):
        """Log text generation samples"""
        if self.writer:
            self.writer.add_text(f"generation/{step}", f"Prompt: {prompt}\nGenerated: {generated}", step)
            
        if self.wandb_run:
            self.wandb_run.log({
                "text_samples": wandb.Html(f"<b>Prompt:</b> {prompt}<br><b>Generated:</b> {generated}")
            }, step=step)

    def close(self):
        """Clean up visualization resources"""
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
