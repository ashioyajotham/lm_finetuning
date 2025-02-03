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
from typing import Dict, Any, Optional, List
import wandb
import os
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

class TrainingVisualizer:
    def __init__(self, training_args, model_name: str, model=None, viz_config=None):
        """Initialize visualization tools based on config"""
        self.training_args = training_args
        self.model_name = model_name
        self.model = model
        self.writer = None
        self.wandb_run = None

        # Create necessary directories
        os.makedirs(training_args.logging_dir, exist_ok=True)
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # Set run name early to ensure consistency
        self.run_name = f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        training_args.run_name = self.run_name  # Override default './results'
        
        # Load environment variables
        load_dotenv()
        
        # Set comprehensive default config
        default_config = {
            'use_tensorboard': True,
            'use_wandb': False,
            'viz_port': 6006  # Add default port
        }
        
        # Update defaults with provided config if any
        self.viz_config = default_config
        if viz_config:
            self.viz_config.update(viz_config)
        
        if self.viz_config['use_tensorboard']:
            try:
                self.writer = SummaryWriter(training_args.logging_dir)
                self._launch_tensorboard()
            except Exception as e:
                print(f"Failed to initialize tensorboard: {e}")
                self.viz_config['use_tensorboard'] = False
            
        if self.viz_config['use_wandb']:
            try:
                # Clear any existing wandb cache
                if os.path.exists(os.path.join(training_args.output_dir, "wandb")):
                    import shutil
                    shutil.rmtree(os.path.join(training_args.output_dir, "wandb"))
                
                self.wandb_run = wandb.init(
                    project=None,
                    entity=None,
                    name=self.run_name,  # Use consistently set run_name
                    dir=training_args.output_dir,
                    job_type='training',
                    config={
                        "model_name": model_name,
                        "training_args": vars(training_args),
                        "viz_config": self.viz_config
                    },
                    settings=wandb.Settings(start_method="thread"),
                    reinit=True
                )
                
                # Ensure wandb uses our run name
                if self.wandb_run:
                    wandb.run.name = self.run_name
                    wandb.run.save()
                    
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
                self.viz_config['use_wandb'] = False

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
        """Log metrics to all active visualization tools"""
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
                
        if self.viz_config['use_wandb'] and self.wandb_run:
            wandb.log(metrics, step=step)

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

    def visualize_attention_patterns(self, layer_idx: int, attention_maps: torch.Tensor):
        """Visualize cross-attention patterns for transformer layers"""
        if self.writer:
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(attention_maps.detach().cpu().numpy(), ax=ax)
            self.writer.add_figure(f'attention_pattern_layer_{layer_idx}', fig)
        if self.wandb_run:
            wandb.log({f'attention_pattern_layer_{layer_idx}': wandb.Image(fig)})

    def log_embedding_space(self, embeddings: torch.Tensor, labels: List[str], step: int):
        """Visualize embedding space using t-SNE"""
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())
        
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        if self.writer:
            self.writer.add_figure('embedding_space', fig, step)
        if self.wandb_run:
            wandb.log({'embedding_space': wandb.Image(fig)}, step=step)

    def log_research_metrics(self, 
                           compression_ratio: float,
                           knowledge_retention: float,
                           fine_tuning_stability: float,
                           step: int):
        """Log research-specific metrics"""
        metrics = {
            'compression_ratio': compression_ratio,
            'knowledge_retention': knowledge_retention,
            'fine_tuning_stability': fine_tuning_stability
        }
        
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f'research_metrics/{name}', value, step)
        if self.wandb_run:
            wandb.log(metrics, step=step)

    def log_layer_gradients(self, named_parameters: Dict[str, torch.Tensor], step: int):
        """Track layer-wise gradient magnitudes"""
        grad_norms = {}
        for name, param in named_parameters:
            if param.grad is not None:
                grad_norms[f'gradient_norm/{name}'] = torch.norm(param.grad).item()
        
        if self.writer:
            for name, norm in grad_norms.items():
                self.writer.add_scalar(name, norm, step)
        if self.wandb_run:
            wandb.log(grad_norms, step=step)

    def close(self):
        """Cleanup visualization resources"""
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()