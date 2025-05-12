#!/usr/bin/env python3
"""
Math Reasoner Demo Script

This script demonstrates the math reasoning capabilities of the system.
Run this script to see the reasoner and critic in action.
"""

import os
import torch
from utils.data import load_aime_dataset
from reasoners.base_reasoner import BaseReasoner
from critics.base_critic import BaseCritic
from reasoners.critic_enhanced_reasoner import CriticEnhancedReasoner
import pandas as pd

def main():
    print("### Math Reasoner Demo ###\n")
    
    # 1. Initialize models
    print("Initializing models...")
    reasoner = BaseReasoner(model_name="meta-llama/Llama-3.1-8B-Instruct")
    critic = BaseCritic(model_name="meta-llama/Llama-3.2-1B")
    enhanced_reasoner = CriticEnhancedReasoner(
        reasoner_model_name="meta-llama/Llama-3.1-8B-Instruct",
        critic_model_name="meta-llama/Llama-3.2-1B"
    )
    
    # 2. Load AIME Dataset
    print("\nLoading AIME dataset...")
    dataset = load_aime_dataset()
    print(f"Dataset size: {len(dataset['train'])} problems")
    
    # 3. Select a problem
    problem_idx = 0  # Change this to explore different problems
    problem = dataset['train'][problem_idx]['Problem']
    solution = dataset['train'][problem_idx]['Solution']
    answer = dataset['train'][problem_idx]['Answer']
    
    print("\n### Problem ###")
    print(problem)
    print("\n### Reference Solution ###")
    print(solution)
    print(f"\nExpected Answer: {answer}")
    
    # 4. Solve with Base Reasoner
    print("\n### Solving with Base Reasoner ###")
    base_solution = reasoner.solve(problem, do_sample=False)[0]
    print(f"\nBase Reasoner Solution:")
    print(base_solution)
    
    # 5. Evaluate with Critic
    print("\n### Critic Evaluation of Base Solution ###")
    evaluation = critic.evaluate(problem, base_solution)
    print(f"Critic Evaluation:")
    print(evaluation['evaluation'])
    print(f"\nScore: {evaluation.get('score', 'Not available')}")
    
    # 6. Solve with Critic-Enhanced Reasoner
    print("\n### Solving with Critic-Enhanced Reasoner ###")
    enhanced_result = enhanced_reasoner.solve(problem, max_iterations=2)
    
    # Display the improvement history
    for i, step in enumerate(enhanced_result["history"]):
        print(f"\n--- Iteration {i} ---")
        if i > 0:
            print("Feedback:")
            print(step["feedback"])
        print("Solution:")
        print(step["solution"])
    
    # 7. Compare solutions
    print("\n### Solution Comparison ###")
    reference_eval = critic.evaluate(problem, solution)
    base_eval = critic.evaluate(problem, base_solution)
    enhanced_eval = critic.evaluate(problem, enhanced_result["final_solution"])
    
    # Show comparison
    print(f"Reference Solution Score: {reference_eval.get('score', 'N/A')}")
    print(f"Base Reasoner Score: {base_eval.get('score', 'N/A')}")
    print(f"Enhanced Reasoner Score: {enhanced_eval.get('score', 'N/A')}")
    
    # 8. Custom problem demo
    print("\n### Custom Problem Demo ###")
    custom_problem = """
    Let O(0,0), A(1/2, 0), and B(0, sqrt(3)/2) be points in the coordinate plane. 
    Let F be the family of segments PQ of unit length lying in the first quadrant with P on the x-axis and Q on the y-axis. 
    There is a unique point C on AB, distinct from A and B, that does not belong to any segment from F other than AB. 
    Then OC^2 = p/q, where p and q are relatively prime positive integers. Find p + q.
    """
    
    print("Custom Problem:")
    print(custom_problem)
    
    print("\nGenerating solutions...")
    custom_base_solution = reasoner.solve(custom_problem, do_sample=False)[0]
    custom_enhanced_result = enhanced_reasoner.solve(custom_problem, max_iterations=2)
    
    print("\nBase Reasoner Solution:")
    print(custom_base_solution)
    
    print("\nEnhanced Reasoner Final Solution:")
    print(custom_enhanced_result["final_solution"])
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 