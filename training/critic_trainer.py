import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from critics.base_critic import BaseCritic
import json
import os
from tqdm import tqdm

class CriticTrainingDataset(Dataset):
    """
    Dataset for training the critic
    """
    def __init__(self, problems, solutions, error_labels, tokenizer, max_length=512):
        """
        Initialize the dataset
        
        Args:
            problems (list): List of problem texts
            solutions (list): List of solution texts
            error_labels (list): List of error annotations 
                                (dict with 'has_error', 'error_indices', 'error_descriptions')
            tokenizer: Tokenizer for encoding the text
            max_length (int): Maximum sequence length
        """
        self.problems = problems
        self.solutions = solutions
        self.error_labels = error_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        solution = self.solutions[idx]
        error_label = self.error_labels[idx]
        
        # Create prompt for critic training
        prompt = f"""You are a mathematical reasoning critic. Your task is to evaluate the following solution to an AIME problem.

Problem:
{problem}

Proposed Solution:
{solution}

Evaluation:
"""
        # Tokenize
        encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                 max_length=self.max_length, padding="max_length")
        
        # Create target continuation based on error labels
        if error_label["has_error"]:
            target_continuation = f"""This solution contains errors:
{error_label['error_descriptions']}

Score: {error_label['score']}/10
"""
        else:
            target_continuation = f"""This solution is correct. All the steps are valid and well-explained.

Score: 10/10
"""
        
        # Tokenize target continuation
        target_encoding = self.tokenizer(target_continuation, return_tensors="pt", truncation=True,
                                        max_length=self.max_length, padding="max_length")
        
        # Combine prompt with target for labels
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = torch.clone(input_ids)
        
        # Set labels for prompt part to -100 (ignored in loss computation)
        labels[:-1] = -100
        
        # Append target continuation tokens to labels (shifted right)
        target_ids = target_encoding["input_ids"].squeeze()[1:]
        labels[-1] = target_ids[0]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "has_error": torch.tensor(1 if error_label["has_error"] else 0)
        }

class CriticTrainer:
    """
    Trainer for the critic model
    """
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", device="cuda"):
        """
        Initialize the trainer
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.device = device
        self.critic = BaseCritic(model_name)
        self.critic.model.to(self.device)
        self.tokenizer = self.critic.tokenizer
        
    def prepare_training_data(self, training_data_path):
        """
        Prepare training data from a JSON file
        
        Args:
            training_data_path (str): Path to JSON file with training data
            
        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        with open(training_data_path, 'r') as f:
            data = json.load(f)
        
        problems = [item["problem"] for item in data]
        solutions = [item["solution"] for item in data]
        error_labels = [item["error_annotation"] for item in data]
        
        # Split into train/val
        train_size = int(0.8 * len(problems))
        
        train_problems = problems[:train_size]
        train_solutions = solutions[:train_size]
        train_error_labels = error_labels[:train_size]
        
        val_problems = problems[train_size:]
        val_solutions = solutions[train_size:]
        val_error_labels = error_labels[train_size:]
        
        # Create datasets
        train_dataset = CriticTrainingDataset(
            train_problems, train_solutions, train_error_labels, self.tokenizer)
        
        val_dataset = CriticTrainingDataset(
            val_problems, val_solutions, val_error_labels, self.tokenizer)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=4)
        
        return train_dataloader, val_dataloader
    
    def train(self, train_dataloader, val_dataloader, num_epochs=3, 
             learning_rate=2e-5, output_dir="critic_model"):
        """
        Train the critic model
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            output_dir (str): Directory to save model checkpoints
            
        Returns:
            dict: Training history
        """
        # Initialize optimizer
        optimizer = optim.AdamW(self.critic.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            self.critic.model.train()
            train_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.critic.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_train_loss = train_loss / len(train_dataloader)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            self.critic.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # Forward pass
                    outputs = self.critic.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            history["val_loss"].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint
            os.makedirs(output_dir, exist_ok=True)
            self.critic.save(f"{output_dir}/epoch_{epoch+1}")
        
        # Save final model
        self.critic.save(output_dir)
        
        return history 