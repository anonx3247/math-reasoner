from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def load_aime_dataset():
    """
    Load the AIME dataset from HuggingFace
    
    Returns:
        datasets.DatasetDict: A dictionary containing the AIME dataset splits
    """
    return load_dataset("Maxwell-Jia/AIME_2024")

class AIMEDataset(Dataset):
    """
    PyTorch Dataset for AIME problems
    """
    def __init__(self, dataset, split='train', tokenizer=None):
        """
        Initialize the AIME dataset
        
        Args:
            dataset: HuggingFace dataset loaded from load_aime_dataset
            split (str): Dataset split to use ('train', 'validation', 'test')
            tokenizer: Tokenizer for encoding the text
        """
        self.data = dataset[split]
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        problem = item['Problem']
        solution = item['Solution']
        answer = item['Answer']
        
        if self.tokenizer:
            problem_encoding = self.tokenizer(problem, return_tensors="pt", padding="max_length", 
                                            truncation=True, max_length=512)
            solution_encoding = self.tokenizer(solution, return_tensors="pt", padding="max_length",
                                             truncation=True, max_length=512)
            
            return {
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "problem_input_ids": problem_encoding["input_ids"].squeeze(),
                "problem_attention_mask": problem_encoding["attention_mask"].squeeze(),
                "solution_input_ids": solution_encoding["input_ids"].squeeze(),
                "solution_attention_mask": solution_encoding["attention_mask"].squeeze(),
            }
        
        return {
            "problem": problem,
            "solution": solution,
            "answer": answer
        }

def get_dataloaders(dataset, tokenizer, batch_size=8, split_ratio=0.8):
    """
    Create DataLoaders for training and validation
    
    Args:
        dataset: HuggingFace dataset loaded from load_aime_dataset
        tokenizer: Tokenizer for encoding the text
        batch_size (int): Batch size for DataLoader
        split_ratio (float): Ratio for train/validation split
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Use the train split since the dataset only has train
    train_dataset = AIMEDataset(dataset, split='train', tokenizer=tokenizer)
    
    # Create train/val split
    train_size = int(len(train_dataset) * split_ratio)
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader 