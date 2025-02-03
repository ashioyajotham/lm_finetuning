import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class EvalVisualizer:
    def __init__(self, save_dir: str = "evaluation_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_category_performance(self, results: dict, title: str) -> str:
        """Plot performance across different categories"""
        plt.figure(figsize=(12, 6))
        categories = []
        scores = []
        
        for key, value in results.items():
            if '_accuracy' in key:
                categories.append(key.replace('_accuracy', ''))
                scores.append(value)
                
        sns.barplot(x=categories, y=scores)
        plt.title(title)
        plt.xticks(rotation=45)
        
        save_path = self.save_dir / f"category_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def plot_perplexity_comparison(self, perplexity_results: dict) -> str:
        """Plot perplexity scores across different text samples"""
        plt.figure(figsize=(10, 6))
        
        sample_ids = list(range(len(perplexity_results['scores'])))
        scores = perplexity_results['scores']
        
        sns.lineplot(x=sample_ids, y=scores, marker='o')
        plt.axhline(y=perplexity_results['mean'], color='r', linestyle='--', 
                   label=f"Mean: {perplexity_results['mean']:.2f}")
        
        plt.title("Perplexity Scores Across Samples")
        plt.xlabel("Sample ID")
        plt.ylabel("Perplexity")
        plt.legend()
        
        save_path = self.save_dir / f"perplexity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path)
        plt.close()
        return str(save_path)

    def generate_report(self, results: dict, template_path: str) -> str:
        """Generate evaluation report from results"""
        with open(template_path) as f:
            template = f.read()
            
        # Create visualizations
        vis_paths = []
        vis_paths.append(self.plot_category_performance(
            results.get('task_specific', {}),
            "Task-Specific Performance"
        ))
        
        if 'perplexity' in results.get('automatic_metrics', {}):
            vis_paths.append(self.plot_perplexity_comparison(
                results['automatic_metrics']['perplexity']
            ))
        
        # Format metrics
        metrics_summary = json.dumps(results.get('automatic_metrics', {}), indent=2)
        
        # Fill template
        report = template.format(
            model_name=results.get('model_name', 'Unknown'),
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            dataset_name=results.get('dataset_name', 'Unknown'),
            metrics_summary=metrics_summary,
            math_metrics=json.dumps(results.get('task_specific', {}).get('math', {}), indent=2),
            logic_metrics=json.dumps(results.get('task_specific', {}).get('logic', {}), indent=2),
            qa_metrics=json.dumps(results.get('task_specific', {}).get('qa', {}), indent=2),
            example_outputs=self._format_examples(results.get('examples', [])),
            human_eval_results=json.dumps(results.get('human_evaluation', {}), indent=2),
            visualization_paths='\n'.join(f"![{i}]({p})" for i, p in enumerate(vis_paths))
        )
        
        # Save report
        report_path = self.save_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
            
        return str(report_path)

    def _format_examples(self, examples: list) -> str:
        """Format example outputs for markdown"""
        if not examples:
            return "No examples available"
            
        formatted = "### Example Generations\n\n"
        for i, example in enumerate(examples, 1):
            formatted += f"#### Example {i}\n"
            formatted += f"**Input:** {example['input']}\n\n"
            formatted += f"**Generated:** {example['generated']}\n\n"
            formatted += "---\n\n"
        return formatted
