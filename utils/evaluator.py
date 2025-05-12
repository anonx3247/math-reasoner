import torch
import json
import os
from tqdm import tqdm
import wandb
from reasoners.base_reasoner import BaseReasoner
from critics.base_critic import BaseCritic
from reasoners.critic_enhanced_reasoner import CriticEnhancedReasoner

class Evaluator:
    """
    Evaluator for assessing model performance on AIME problems
    """
    def __init__(self, dataset, eval_type="human"):
        """
        Initialize the evaluator
        
        Args:
            dataset: HuggingFace dataset with AIME problems
            eval_type (str): Type of evaluator to use ('human' or 'gpt4')
        """
        self.dataset = dataset
        self.eval_type = eval_type
        
    def evaluate_reasoner(self, reasoner, output_path=None):
        """
        Evaluate a reasoner model on the dataset
        
        Args:
            reasoner: Reasoner model to evaluate
            output_path (str, optional): Path to save evaluation results
            
        Returns:
            dict: Evaluation results
        """
        results = []
        
        # Get problems and expected answers from dataset
        problems = [item["Problem"] for item in self.dataset["train"]]
        expected_answers = [item["Answer"] for item in self.dataset["train"]]
        
        # Setup tracking
        if self.eval_type != "human":
            wandb.init(project="math-reasoner", name="reasoner-evaluation")
        
        correct_count = 0
        
        # Evaluate each problem
        for i, (problem, expected) in enumerate(tqdm(zip(problems, expected_answers), total=len(problems))):
            # Generate solution
            solutions = reasoner.solve(problem, do_sample=False)
            solution = solutions[0]
            
            # Extract answer (simple parsing, can be improved)
            try:
                # Look for the answer format
                if "\\boxed{" in solution:
                    answer_part = solution.split("\\boxed{")[1].split("}")[0]
                    # Try to extract a number
                    answer = int(''.join(c for c in answer_part if c.isdigit()))
                else:
                    # Look for "Answer:" or similar
                    for line in solution.split('\n'):
                        if "answer" in line.lower() and any(c.isdigit() for c in line):
                            answer = int(''.join(c for c in line if c.isdigit()))
                            break
                    else:
                        answer = None
            except:
                answer = None
            
            # Check if correct
            is_correct = answer is not None and answer == expected
            
            if is_correct:
                correct_count += 1
            
            # Save result
            result = {
                "problem_id": i,
                "problem": problem,
                "solution": solution,
                "extracted_answer": answer,
                "expected_answer": expected,
                "is_correct": is_correct
            }
            
            results.append(result)
            
            # Log to wandb
            if self.eval_type != "human":
                wandb.log({
                    "problem_id": i,
                    "is_correct": is_correct,
                    "running_accuracy": correct_count / (i + 1)
                })
        
        # Calculate overall accuracy
        accuracy = correct_count / len(problems)
        
        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    "results": results,
                    "accuracy": accuracy
                }, f, indent=2)
        
        # Close wandb
        if self.eval_type != "human":
            wandb.finish()
        
        return {
            "results": results,
            "accuracy": accuracy
        }
    
    def evaluate_critic(self, critic, output_path=None):
        """
        Evaluate a critic model on the dataset
        
        Args:
            critic: Critic model to evaluate
            output_path (str, optional): Path to save evaluation results
            
        Returns:
            dict: Evaluation results
        """
        results = []
        
        # Get problems and solutions from dataset
        problems = [item["Problem"] for item in self.dataset["train"]]
        solutions = [item["Solution"] for item in self.dataset["train"]]
        expected_answers = [item["Answer"] for item in self.dataset["train"]]
        
        # Setup tracking
        if self.eval_type != "human":
            wandb.init(project="math-reasoner", name="critic-evaluation")
        
        # Evaluate each solution
        for i, (problem, solution, expected) in enumerate(tqdm(zip(problems, solutions, expected_answers), total=len(problems))):
            # Get critic evaluation
            evaluation = critic.evaluate(problem, solution)
            
            # Save result
            result = {
                "problem_id": i,
                "problem": problem,
                "solution": solution,
                "expected_answer": expected,
                "critic_evaluation": evaluation["evaluation"],
                "critic_score": evaluation["score"]
            }
            
            results.append(result)
            
            # Log to wandb
            if self.eval_type != "human":
                wandb.log({
                    "problem_id": i,
                    "critic_score": evaluation["score"]
                })
        
        # Calculate average score
        avg_score = sum(r["critic_score"] for r in results if r["critic_score"] is not None) / len(results)
        
        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    "results": results,
                    "average_score": avg_score
                }, f, indent=2)
        
        # Close wandb
        if self.eval_type != "human":
            wandb.finish()
        
        return {
            "results": results,
            "average_score": avg_score
        }
    
    def evaluate_with_human_feedback(self, reasoner, output_path=None):
        """
        Generate solutions for human evaluation
        
        Args:
            reasoner: Reasoner model to evaluate
            output_path (str): Path to save solutions for human evaluation
            
        Returns:
            dict: Generated solutions
        """
        if self.eval_type != "human":
            raise ValueError("This method is only for human evaluation")
        
        # Get problems from dataset
        problems = [item["Problem"] for item in self.dataset["train"]]
        expected_answers = [item["Answer"] for item in self.dataset["train"]]
        
        solutions = []
        
        # Generate solutions
        for i, (problem, expected) in enumerate(tqdm(zip(problems, expected_answers), total=len(problems))):
            # Generate solution
            solution_text = reasoner.solve(problem, do_sample=False)[0]
            
            # Save result
            solution = {
                "problem_id": i,
                "problem": problem,
                "solution": solution_text,
                "expected_answer": expected,
                "human_evaluation": {
                    "score": None,
                    "feedback": None,
                    "error_annotation": None
                }
            }
            
            solutions.append(solution)
        
        # Save solutions for human evaluation
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    "solutions": solutions
                }, f, indent=2)
        
        return {
            "solutions": solutions
        } 