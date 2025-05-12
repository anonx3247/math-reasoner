import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseReasoner:
    """
    Base Reasoner model using Llama-3.1-8B-Instruct
    """
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the base reasoner model
        
        Args:
            model_name (str): Hugging Face model name
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def solve(self, problem, max_new_tokens=512, do_sample=True, top_k=5, num_return_sequences=1):
        """
        Solve a math problem
        
        Args:
            problem (str): The math problem to solve
            max_new_tokens (int): Maximum number of new tokens to generate
            do_sample (bool): Whether to use sampling
            top_k (int): Top-k sampling parameter
            num_return_sequences (int): Number of solutions to generate
            
        Returns:
            list: Generated solutions
        """
        prompt = f"""You are a mathematical reasoning expert assistant. Your task is to solve the following AIME (American Invitational Mathematics Examination) problem step by step.
        
Problem:
{problem}

Solution:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        solutions = []
        for output in outputs:
            solution = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt part
            solution = solution.replace(prompt, "").strip()
            solutions.append(solution)
        
        return solutions
    
    def get_top_k_logits(self, text, k=5):
        """
        Get the top-k logits for the next token
        
        Args:
            text (str): Input text
            k (int): Number of top logits to return
            
        Returns:
            tuple: (top_k_logits, top_k_tokens)
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
            
        # Get logits for the next token
        next_token_logits = outputs.logits[:, -1, :]
        
        # Get top k logits and their corresponding token ids
        top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
        
        # Convert token ids to tokens
        top_k_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices[0]]
        
        return top_k_logits[0].tolist(), top_k_tokens
    
    def save(self, path):
        """Save the model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path):
        """Load the model and tokenizer"""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path) 