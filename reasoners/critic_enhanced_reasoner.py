import torch
import torch.nn as nn
from reasoners.base_reasoner import BaseReasoner
from critics.base_critic import BaseCritic

class CriticEnhancedReasoner:
    """
    Reasoner model enhanced with critic feedback during inference
    """
    def __init__(self, 
                 reasoner_model_name="meta-llama/Llama-3.1-8B-Instruct",
                 critic_model_name="meta-llama/Llama-3.2-1B"):
        """
        Initialize the critic-enhanced reasoner
        
        Args:
            reasoner_model_name (str): Hugging Face model name for reasoner
            critic_model_name (str): Hugging Face model name for critic
        """
        self.reasoner = BaseReasoner(reasoner_model_name)
        self.critic = BaseCritic(critic_model_name)
    
    def solve(self, problem, max_new_tokens=512, max_iterations=3, temperature=1.0):
        """
        Solve a math problem with critic feedback
        
        Args:
            problem (str): The math problem to solve
            max_new_tokens (int): Maximum number of new tokens per generation
            max_iterations (int): Maximum number of reasoning iterations
            temperature (float): Temperature for token sampling
            
        Returns:
            dict: Final solution and improvement history
        """
        history = []
        
        # Initial solution
        solutions = self.reasoner.solve(
            problem, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            top_k=5,
            num_return_sequences=1
        )
        
        current_solution = solutions[0]
        history.append({
            "solution": current_solution,
            "iteration": 0,
            "feedback": None
        })
        
        # Iterative improvement
        for i in range(max_iterations):
            # Get critic feedback
            evaluation = self.critic.evaluate(problem, current_solution)
            
            # Create improved prompt based on feedback
            improved_prompt = f"""You are a mathematical reasoning expert assistant. Your task is to solve the following AIME (American Invitational Mathematics Examination) problem step by step.
            
Problem:
{problem}

Your previous solution:
{current_solution}

Critique of your solution:
{evaluation['evaluation']}

Please provide an improved solution addressing the above feedback:
"""
            
            # Generate improved solution
            inputs = self.reasoner.tokenizer(improved_prompt, return_tensors="pt")
            
            outputs = self.reasoner.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=5,
                num_return_sequences=1,
                pad_token_id=self.reasoner.tokenizer.eos_token_id
            )
            
            improved_solution = self.reasoner.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the improved solution part
            improved_solution = improved_solution.split("Please provide an improved solution addressing the above feedback:")[1].strip()
            
            # Update current solution
            current_solution = improved_solution
            
            # Add to history
            history.append({
                "solution": current_solution,
                "iteration": i + 1,
                "feedback": evaluation['evaluation']
            })
            
            # Check if score is high enough to stop
            if evaluation.get('score', 0) >= 8:
                break
        
        return {
            "final_solution": current_solution,
            "history": history
        }
    
    def solve_step_by_step(self, problem, max_new_tokens=512, max_steps=5, temperature=1.0):
        """
        Generate a solution token by token, with critic influence at each step
        
        Args:
            problem (str): The math problem to solve
            max_new_tokens (int): Maximum number of tokens to generate
            max_steps (int): Maximum number of generation steps
            temperature (float): Temperature for sampling
            
        Returns:
            str: Generated solution
        """
        prompt = f"""You are a mathematical reasoning expert assistant. Your task is to solve the following AIME (American Invitational Mathematics Examination) problem step by step.
        
Problem:
{problem}

Solution:"""
        
        solution = prompt
        
        for _ in range(max_new_tokens):
            # Get reasoner token probabilities
            reasoner_logits, reasoner_tokens = self.reasoner.get_top_k_logits(solution, k=10)
            
            # Get critic probabilities for the same tokens
            critic_probs = []
            for token in reasoner_tokens:
                # Evaluate each potential token
                potential_text = solution + token
                inputs = self.critic.tokenizer(potential_text, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.critic.model(inputs["input_ids"])
                    logits = outputs.logits[:, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Higher values indicate critic prefers this token
                    score = log_probs.max().item()
                    critic_probs.append(score)
            
            # Combine reasoner and critic scores
            combined_scores = []
            for reasoner_score, critic_score in zip(reasoner_logits, critic_probs):
                combined_scores.append(reasoner_score + critic_score)
                
            # Choose token with highest combined score
            best_idx = combined_scores.index(max(combined_scores))
            next_token = reasoner_tokens[best_idx]
            
            solution += next_token
            
            # Check for end of generation
            if next_token == self.reasoner.tokenizer.eos_token or "Answer:" in solution:
                break
        
        # Remove the prompt part
        solution = solution.replace(prompt, "").strip()
        
        return solution
    
    def save(self, reasoner_path, critic_path):
        """Save both models"""
        self.reasoner.save(reasoner_path)
        self.critic.save(critic_path)
    
    def load(self, reasoner_path, critic_path):
        """Load both models"""
        self.reasoner.load(reasoner_path)
        self.critic.load(critic_path) 