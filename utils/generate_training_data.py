import json
import os
import argparse
from tqdm import tqdm
import random
from utils.data import load_aime_dataset
from reasoners.base_reasoner import BaseReasoner
from critics.base_critic import BaseCritic
import torch

def generate_synthetic_critic_data(reasoner, dataset, num_samples=30, output_path="data/critic_training_data.json"):
    """
    Generate synthetic training data for the critic by having the reasoner solve problems
    and then injecting errors into some solutions
    
    Args:
        reasoner: BaseReasoner instance
        dataset: HuggingFace dataset with AIME problems
        num_samples (int): Number of samples to generate
        output_path (str): Path to save generated data
        
    Returns:
        list: Generated training data
    """
    training_data = []
    
    # Get problems from dataset
    problems = [item["Problem"] for item in dataset["train"]]
    expected_answers = [item["Answer"] for item in dataset["train"]]
    
    # Sample problems
    if num_samples > len(problems):
        # Use all problems and repeat some
        sample_indices = list(range(len(problems))) + random.sample(range(len(problems)), num_samples - len(problems))
    else:
        # Randomly sample problems
        sample_indices = random.sample(range(len(problems)), num_samples)
    
    for idx in tqdm(sample_indices, desc="Generating critic training data"):
        problem = problems[idx]
        expected_answer = expected_answers[idx]
        
        # Generate a solution
        solution = reasoner.solve(problem, do_sample=True)[0]
        
        # Decide whether to add errors (50% chance)
        has_error = random.random() < 0.5
        
        if has_error:
            # Add some synthetic errors
            error_type = random.choice(["calculation", "reasoning", "skip_step"])
            
            if error_type == "calculation":
                # Replace a number with a wrong one
                numbers = []
                for i, char in enumerate(solution):
                    if char.isdigit() and (i == 0 or not solution[i-1].isdigit()):
                        numbers.append(i)
                
                if numbers:
                    pos = random.choice(numbers)
                    original_digit = solution[pos]
                    new_digit = str(random.choice([d for d in range(10) if str(d) != original_digit]))
                    solution = solution[:pos] + new_digit + solution[pos+1:]
                    
                    error_description = f"There is a calculation error. The digit {original_digit} should be {new_digit}."
                    error_indices = [pos]
                
            elif error_type == "reasoning":
                # Add an incorrect reasoning step
                lines = solution.split("\n")
                if len(lines) > 3:
                    line_idx = random.randint(1, len(lines) - 2)
                    original_line = lines[line_idx]
                    lines[line_idx] = "This step has incorrect reasoning: " + original_line
                    solution = "\n".join(lines)
                    
                    error_description = f"There is a reasoning error in step {line_idx+1}."
                    error_indices = [sum(len(l) + 1 for l in lines[:line_idx])]
                
            elif error_type == "skip_step":
                # Remove a step
                lines = solution.split("\n")
                if len(lines) > 4:
                    line_idx = random.randint(1, len(lines) - 3)
                    removed_line = lines.pop(line_idx)
                    solution = "\n".join(lines)
                    
                    error_description = f"A step was skipped: {removed_line}"
                    error_indices = [sum(len(l) + 1 for l in lines[:line_idx])]
            
            # Random score between 3 and 7
            score = random.randint(3, 7)
            
            error_annotation = {
                "has_error": True,
                "error_type": error_type,
                "error_indices": error_indices,
                "error_descriptions": error_description,
                "score": score
            }
        else:
            # No errors
            error_annotation = {
                "has_error": False,
                "error_type": None,
                "error_indices": [],
                "error_descriptions": "",
                "score": 10
            }
        
        # Add to training data
        training_data.append({
            "problem": problem,
            "solution": solution,
            "expected_answer": expected_answer,
            "error_annotation": error_annotation
        })
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)
    
    return training_data

def generate_synthetic_reasoner_dpo_data(reasoner, critic, dataset, num_samples=30, output_path="data/reasoner_dpo_data.json"):
    """
    Generate synthetic training data for DPO training of the reasoner
    
    Args:
        reasoner: BaseReasoner instance
        critic: BaseCritic instance
        dataset: HuggingFace dataset with AIME problems
        num_samples (int): Number of samples to generate
        output_path (str): Path to save generated data
        
    Returns:
        list: Generated training data
    """
    training_data = []
    
    # Get problems from dataset
    problems = [item["Problem"] for item in dataset["train"]]
    solutions = [item["Solution"] for item in dataset["train"]]
    
    # Sample problems
    if num_samples > len(problems):
        # Use all problems and repeat some
        sample_indices = list(range(len(problems))) + random.sample(range(len(problems)), num_samples - len(problems))
    else:
        # Randomly sample problems
        sample_indices = random.sample(range(len(problems)), num_samples)
    
    for idx in tqdm(sample_indices, desc="Generating DPO training data"):
        problem = problems[idx]
        reference_solution = solutions[idx]
        
        # Generate multiple solutions
        sample_solutions = reasoner.solve(problem, do_sample=True, num_return_sequences=3)
        
        # Also add the reference solution
        all_solutions = sample_solutions + [reference_solution]
        
        # Have critic evaluate all solutions
        solution_data = []
        for solution in all_solutions:
            evaluation = critic.evaluate(problem, solution)
            score = evaluation.get("score", 5)  # Default to 5 if no score
            
            solution_data.append({
                "solution_text": solution,
                "critic_evaluation": evaluation["evaluation"],
                "critic_score": score,
                "is_correct": score >= 8  # Treat solutions with score 8+ as correct
            })
        
        # Add to training data
        training_data.append({
            "problem": problem,
            "solutions": solution_data
        })
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)
    
    return training_data

def main():
    parser = argparse.ArgumentParser(description="Generate training data for Math Reasoner")
    
    parser.add_argument("--data_type", type=str, required=True, 
                       choices=["critic", "reasoner_dpo"], 
                       help="Type of training data to generate")
    
    parser.add_argument("--reasoner_model", type=str, 
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Path to reasoner model or HF model name")
    
    parser.add_argument("--critic_model", type=str, 
                       default="meta-llama/Llama-3.2-1B",
                       help="Path to critic model or HF model name")
    
    parser.add_argument("--num_samples", type=int, default=30,
                       help="Number of samples to generate")
    
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save generated data")
    
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_aime_dataset()
    
    # Load models
    if args.data_type == "critic" or args.data_type == "reasoner_dpo":
        reasoner = BaseReasoner(args.reasoner_model)
    
    if args.data_type == "reasoner_dpo":
        critic = BaseCritic(args.critic_model)
    
    # Generate data
    if args.data_type == "critic":
        generate_synthetic_critic_data(
            reasoner=reasoner,
            dataset=dataset,
            num_samples=args.num_samples,
            output_path=args.output_path
        )
    elif args.data_type == "reasoner_dpo":
        generate_synthetic_reasoner_dpo_data(
            reasoner=reasoner,
            critic=critic,
            dataset=dataset,
            num_samples=args.num_samples,
            output_path=args.output_path
        )
    
    print(f"Generated {args.num_samples} samples and saved to {args.output_path}")

if __name__ == "__main__":
    main() 