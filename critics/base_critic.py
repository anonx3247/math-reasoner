import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseCritic:
    """
    Base Critic model using Llama-3.2-1B
    """
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        """
        Initialize the base critic model
        
        Args:
            model_name (str): Hugging Face model name
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def evaluate(self, problem, solution, max_new_tokens=256):
        """
        Evaluate a solution to a math problem
        
        Args:
            problem (str): The math problem
            solution (str): The solution to evaluate
            max_new_tokens (int): Maximum number of new tokens to generate
            
        Returns:
            dict: Evaluation results with scores and feedback
        """
        prompt = f"""You are a mathematical reasoning critic. Your task is to evaluate the following solution to an AIME (American Invitational Mathematics Examination) problem. 
        
Problem:
{problem}

Proposed Solution:
{solution}

Evaluation:
- Check the solution step by step
- Identify any errors in reasoning or calculations
- Score the solution's validity (0-10)
- Provide constructive feedback

"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate evaluation
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for evaluation
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt part
        evaluation = evaluation.replace(prompt, "").strip()
        
        # Extract score (simple parsing, can be improved)
        try:
            score_line = [line for line in evaluation.split('\n') if "Score" in line][0]
            score = float(score_line.split(':')[1].strip().split('/')[0])
        except:
            score = None
        
        return {
            "evaluation": evaluation,
            "score": score
        }
    
    def token_probabilities(self, text, token_ids=None):
        """
        Calculate probabilities for each token
        
        Args:
            text (str): Input text
            token_ids (list, optional): List of token IDs to get probabilities for
            
        Returns:
            list: Token probabilities
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
            
        logits = outputs.logits
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        if token_ids is None:
            # Return probabilities for all tokens in the sequence
            return probabilities.squeeze().tolist()
        else:
            # Return probabilities for specific token IDs
            return [probabilities[0, i, token_id].item() for i, token_id in enumerate(token_ids)]
    
    def save(self, path):
        """Save the model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path):
        """Load the model and tokenizer"""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path) 