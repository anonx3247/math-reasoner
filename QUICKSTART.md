# Math Reasoner - Quick Start Guide

This guide will help you get started with the Math Reasoner project, a system for solving AIME (American Invitational Mathematics Examination) problems using language models.

## Setup

1. Clone the repository and install the requirements:

```bash
git clone https://github.com/yourusername/math-reasoner.git
cd math-reasoner
pip install -e .
```

2. Make sure you have access to the required language models:
   - Reasoner: `meta-llama/Llama-3.1-8B-Instruct`
   - Critic: `meta-llama/Llama-3.2-1B`

## Basic Usage

### Inference

To solve a math problem using the base reasoner:

```bash
python main.py --mode inference --problem "Let x,y and z be positive real numbers that satisfy the following system of equations: log_2(x/yz) = 1/2, log_2(y/xz) = 1/3, log_2(z/xy) = 1/4. Then the value of |log_2(x^4y^3z^2)| is m/n where m and n are relatively prime positive integers. Find m+n."
```

To use the critic-enhanced reasoner for better solutions:

```bash
python main.py --mode inference --problem "Let x,y and z be positive real numbers that satisfy the following system of equations: log_2(x/yz) = 1/2, log_2(y/xz) = 1/3, log_2(z/xy) = 1/4. Then the value of |log_2(x^4y^3z^2)| is m/n where m and n are relatively prime positive integers. Find m+n." --use_critic
```

### Evaluation

To evaluate a reasoner on the AIME dataset:

```bash
python main.py --mode evaluate_reasoner --reasoner_model "meta-llama/Llama-3.1-8B-Instruct" --output_dir "outputs"
```

To evaluate a critic on the AIME dataset:

```bash
python main.py --mode evaluate_critic --critic_model "meta-llama/Llama-3.2-1B" --output_dir "outputs"
```

## Training

### Generating Training Data

First, generate synthetic training data:

```bash
# Generate critic training data
python -m utils.generate_training_data --data_type critic --num_samples 50 --output_path "data/critic_training_data.json"

# Generate reasoner DPO training data
python -m utils.generate_training_data --data_type reasoner_dpo --num_samples 50 --output_path "data/reasoner_dpo_data.json"
```

### Training the Critic

```bash
python main.py --mode train_critic --critic_model "meta-llama/Llama-3.2-1B" --training_data "data/critic_training_data.json" --num_epochs 3 --output_dir "outputs"
```

### Training the Reasoner

#### Using Critic Feedback

```bash
python main.py --mode train_reasoner --reasoner_model "meta-llama/Llama-3.1-8B-Instruct" --critic_model "outputs/critic_model" --num_epochs 3 --output_dir "outputs"
```

#### Using DPO (Direct Preference Optimization)

```bash
python main.py --mode train_reasoner_dpo --reasoner_model "meta-llama/Llama-3.1-8B-Instruct" --training_data "data/reasoner_dpo_data.json" --num_epochs 3 --output_dir "outputs"
```

## Project Structure

- `main.py`: Main entry point for the application
- `model.py`: Original model implementation
- `reasoners/`: Contains reasoner model implementations
  - `base_reasoner.py`: Basic reasoner using Llama-3.1-8B-Instruct
  - `critic_enhanced_reasoner.py`: Reasoner enhanced with critic feedback
- `critics/`: Contains critic model implementations
  - `base_critic.py`: Basic critic using Llama-3.2-1B
- `training/`: Contains training modules
  - `critic_trainer.py`: Trainer for the critic model
  - `reasoner_trainer.py`: Trainer for the reasoner model
- `utils/`: Utility functions
  - `data.py`: Data loading and processing utilities
  - `evaluator.py`: Model evaluation utilities
  - `generate_training_data.py`: Training data generation
- `data/`: Directory for storing datasets and training data
- `outputs/`: Directory for storing outputs, model checkpoints

## Extending the Project

### Adding a New Reasoner

1. Create a new file in the `reasoners/` directory
2. Implement your reasoner class
3. Use the new reasoner in `main.py`

### Adding a New Critic

1. Create a new file in the `critics/` directory
2. Implement your critic class 
3. Use the new critic in `main.py`

### Using Different Training Methods

Modify the trainers in the `training/` directory to implement custom training approaches.

## References

- [AIME_2024 Dataset](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024)
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) 