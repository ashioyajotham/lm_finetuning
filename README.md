# Language Model Fine-tuning Project

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python train.py
```

## Configuration

Edit `config.py` to modify:
- Model selection (default: "gpt2")
- Training parameters
- Dataset selection

## Available Datasets

Some recommended datasets from Hugging Face:
- "wikitext" (wikipedia text)
- "bookcorpus"
- "squad" (question-answering)
- "glue"

## Example Configuration

To use a different dataset, modify `config.py`:
```python
@dataclass
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    max_length: int = 128
    train_test_split: float = 0.1

@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    batch_size: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    logging_steps: int = 100
    save_steps: int = 1000
    output_dir: str = "output"
```
