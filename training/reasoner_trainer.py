import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from reasoners.base_reasoner import BaseReasoner
from critics.base_critic import BaseCritic
from trl import DPOTrainer
import json
import os
from tqdm import tqdm
import wandb

class ReasonerTrainingDataset(Dataset):
    """
    Dataset for training the reasoner model
    """
    def __init__(self, problems, correct_solutions, incorrect_solutions, tokenizer, max_length=512):
        """
        Initialize the dataset
        
        Args:
            problems (list): List of problem texts
            correct_solutions (list): List of correct solution texts
            incorrect_solutions (list): List of incorrect solution texts
            tokenizer: Tokenizer for encoding the text
            max_length (int): Maximum sequence length
        """
        self.problems = problems
        self.correct_solutions = correct_solutions
        self.incorrect_solutions = incorrect_solutions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        correct_solution = self.correct_solutions[idx]
        incorrect_solution = self.incorrect_solutions[idx]
        
        # Create prompt
        prompt = f"""You are a mathematical reasoning expert assistant. Your task is to solve the following AIME (American Invitational Mathematics Examination) problem step by step.
        
Problem:
{problem}

Solution:"""
        
        # Create encoding for prompt
        prompt_encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                       max_length=self.max_length, padding="max_length")
        
        # Create encodings for solutions
        correct_encoding = self.tokenizer(correct_solution, return_tensors="pt", truncation=True,
                                        max_length=self.max_length, padding="max_length")
        
        incorrect_encoding = self.tokenizer(incorrect_solution, return_tensors="pt", truncation=True,
                                          max_length=self.max_length, padding="max_length")
        
        return {
            "prompt": prompt,
            "prompt_input_ids": prompt_encoding["input_ids"].squeeze(),
            "prompt_attention_mask": prompt_encoding["attention_mask"].squeeze(),
            "correct_input_ids": correct_encoding["input_ids"].squeeze(),
            "correct_attention_mask": correct_encoding["attention_mask"].squeeze(),
            "incorrect_input_ids": incorrect_encoding["input_ids"].squeeze(),
            "incorrect_attention_mask": incorrect_encoding["attention_mask"].squeeze(),
        }

class ReasonerTrainer:
    """
    Trainer for the reasoner model
    """
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
        """
        Initialize the trainer
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.device = device
        self.reasoner = BaseReasoner(model_name)
        self.reasoner.model.to(self.device)
        self.tokenizer = self.reasoner.tokenizer
        
    def prepare_dpo_training_data(self, training_data_path):
        """
        Prepare training data for DPO from a JSON file
        
        Args:
            training_data_path (str): Path to JSON file with training data
            
        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        with open(training_data_path, 'r') as f:
            data = json.load(f)
        
        problems = []
        correct_solutions = []
        incorrect_solutions = []
        
        for item in data:
            problems.append(item["problem"])
            
            # Get correct and incorrect solutions
            solutions = item["solutions"]
            correct = [sol for sol in solutions if sol["is_correct"]]
            incorrect = [sol for sol in solutions if not sol["is_correct"]]
            
            # If we have at least one of each, add to dataset
            if correct and incorrect:
                # Use the first correct and incorrect solutions
                correct_solutions.append(correct[0]["solution_text"])
                incorrect_solutions.append(incorrect[0]["solution_text"])
        
        # Split into train/val
        train_size = int(0.8 * len(problems))
        
        train_problems = problems[:train_size]
        train_correct = correct_solutions[:train_size]
        train_incorrect = incorrect_solutions[:train_size]
        
        val_problems = problems[train_size:]
        val_correct = correct_solutions[train_size:]
        val_incorrect = incorrect_solutions[train_size:]
        
        # Create datasets
        train_dataset = ReasonerTrainingDataset(
            train_problems, train_correct, train_incorrect, self.tokenizer)
        
        val_dataset = ReasonerTrainingDataset(
            val_problems, val_correct, val_incorrect, self.tokenizer)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=2)
        
        return train_dataloader, val_dataloader
    
    def train_dpo(self, train_dataloader, val_dataloader, num_epochs=3, 
                learning_rate=5e-7, beta=0.1, output_dir="reasoner_model"):
        """
        Train the reasoner model using Direct Preference Optimization (DPO)
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            beta (float): DPO beta parameter
            output_dir (str): Directory to save model checkpoints
            
        Returns:
            dict: Training history
        """
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.reasoner.model,
            ref_model=self.reasoner.model,
            beta=beta,
            train_dataset=train_dataloader.dataset,
            eval_dataset=val_dataloader.dataset,
            tokenizer=self.tokenizer,
            max_length=512,
            max_prompt_length=256,
            optim="adamw_torch",
            learning_rate=learning_rate,
            output_dir=output_dir,
        )
        
        # Train the model
        dpo_trainer.train()
        
        # Save final model
        dpo_trainer.save_model(output_dir)
        
        # Update our model
        self.reasoner.model = dpo_trainer.model
        
        return {"trained": True}
    
    def train_with_critic_feedback(self, dataset, critic, num_epochs=3, 
                                 learning_rate=2e-5, output_dir="reasoner_model"):
        """
        Train the reasoner model using feedback from a critic
        
        Args:
            dataset: HuggingFace dataset with AIME problems
            critic: Critic model instance
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            output_dir (str): Directory to save model checkpoints
            
        Returns:
            dict: Training history
        """
        # Initialize optimizer
        optimizer = optim.AdamW(self.reasoner.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            "reward_scores": []
        }
        
        # Use wandb for tracking
        wandb.init(project="math-reasoner", name="reasoner-rl-training")
        
        # Get problems from dataset
        problems = [item["Problem"] for item in dataset["train"]]
        
        # Training loop
        for epoch in range(num_epochs):
            total_reward = 0
            
            for problem in tqdm(problems, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Generate a solution
                solution = self.reasoner.solve(problem, do_sample=True)[0]
                
                # Get critic evaluation
                evaluation = critic.evaluate(problem, solution)
                reward = evaluation.get('score', 0)
                
                # Create improved prompt based on feedback
                improved_prompt = f"""You are a mathematical reasoning expert assistant. Your task is to solve the following AIME problem step by step.
                
Problem:
{problem}

Critique of previous solution:
{evaluation['evaluation']}

Solution:"""
                
                # Generate improved solution (using the model's own output as training signal)
                inputs = self.tokenizer(improved_prompt, return_tensors="pt").to(self.device)
                
                # Forward pass
                outputs = self.reasoner.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["input_ids"]  # Use input as target for teacher forcing
                )
                
                loss = outputs.loss
                
                # Backward pass weighted by reward
                scaled_loss = loss * (10 - reward) / 10  # Scale loss by inverse reward
                
                optimizer.zero_grad()
                scaled_loss.backward()
                optimizer.step()
                
                total_reward += reward
                
                # Log to wandb
                wandb.log({
                    "loss": loss.item(),
                    "scaled_loss": scaled_loss.item(),
                    "reward": reward
                })
            
            avg_reward = total_reward / len(problems)
            history["reward_scores"].append(avg_reward)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Average Reward: {avg_reward:.4f}")
            
            # Save checkpoint
            os.makedirs(output_dir, exist_ok=True)
            self.reasoner.save(f"{output_dir}/epoch_{epoch+1}")
        
        # Save final model
        self.reasoner.save(output_dir)
        
        # Close wandb
        wandb.finish()
        
        return history
    
    def save(self, path):
        """Save the model"""
        self.reasoner.save(path)
    
    def load(self, path):
        """Load the model"""
        self.reasoner.load(path) 