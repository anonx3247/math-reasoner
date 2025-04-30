# Math Reasoner

A project exploring the use of language models for mathematical reasoning, with a focus on solving AIME (American Invitational Mathematics Examination) problems.

## Authors
- Anas Lecaillon ([@anonx3247](https://github.com/anonx3247))
- Timothée Colette ([@lucelsnad](https://github.com/lucelsnad))

## Architecture

The system consists of four main components:

- **Reasoner (R)**: Llama-3.1-8B-Instruct
- **Critic (C)**: Llama-3.2-1B
- **Dataset (D)**: AIME_2024
- **Evaluator (E)**: Human or GPT-4

## Training Procedure

### Critic Training

1. **Human Training**
   - Iterate through AIME problems solved by the Reasoner
   - Human evaluators mark errors in reasoning
   - Critic is trained to assign Pr() = 0 to problematic tokens

2. **Synthetic Training (Alternative)**
   - Use a large reasoner (e.g., GPT-4) to identify errors in Reasoner solutions
   - Trainer has access to solved problems
   - Critic is trained on evaluator outputs

### Reasoner Training

1. **Reasoning Pre-Training**
   - DPO training on AIME problem-solving
   - Bad examples sourced from Critic training data

2. **RL-Training**
   - Critic evaluates Reasoner outputs
   - Reinforcement learning improves reward acquisition

## Reasoner Architectures

1. **Base Reasoner**
   - Trained using the standard procedure

2. **Test-time Critic-enabled Reasoner**
   - Output distribution influenced by Critic at inference time
   - Can be tested on both trained and untrained reasoners

## Project Goals

- Develop robust mathematical reasoning capabilities in language models
- Explore the effectiveness of critic-based training approaches
- Create a system that can solve complex mathematical problems with high accuracy
- Investigate the impact of different training methodologies on model performance
