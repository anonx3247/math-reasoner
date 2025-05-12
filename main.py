import argparse
import os
import torch
from utils.data import load_aime_dataset, get_dataloaders
from reasoners.base_reasoner import BaseReasoner
from critics.base_critic import BaseCritic
from reasoners.critic_enhanced_reasoner import CriticEnhancedReasoner
from training.critic_trainer import CriticTrainer
from training.reasoner_trainer import ReasonerTrainer
from utils.evaluator import Evaluator

def setup_argparse():
    parser = argparse.ArgumentParser(description="Math Reasoner System")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["train_critic", "train_reasoner", "train_reasoner_dpo", 
                               "evaluate_reasoner", "evaluate_critic", "inference"],
                       help="Operating mode")
    
    # Model paths
    parser.add_argument("--reasoner_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Path to reasoner model or HF model name")
    parser.add_argument("--critic_model", type=str, default="meta-llama/Llama-3.2-1B",
                       help="Path to critic model or HF model name")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    
    # Data parameters
    parser.add_argument("--training_data", type=str, 
                       help="Path to training data (for critic or DPO training)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory to save outputs")
    
    # Inference parameters
    parser.add_argument("--problem", type=str,
                       help="Problem to solve in inference mode")
    parser.add_argument("--use_critic", action="store_true",
                       help="Use critic for inference")
    
    # Device selection
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda or cpu)")
    
    return parser.parse_args()

def train_critic(args):
    print("Training Critic...")
    
    # Initialize trainer
    trainer = CriticTrainer(model_name=args.critic_model, device=args.device)
    
    # Prepare training data
    train_dataloader, val_dataloader = trainer.prepare_training_data(args.training_data)
    
    print(f"Training data: {len(train_dataloader.dataset)} examples")
    print(f"Validation data: {len(val_dataloader.dataset)} examples")
    
    # Train model
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=os.path.join(args.output_dir, "critic_model")
    )
    
    print("Critic training completed")
    return history

def train_reasoner_with_critic(args):
    print("Training Reasoner with Critic feedback...")
    
    # Load dataset
    dataset = load_aime_dataset()
    
    # Initialize models
    reasoner_trainer = ReasonerTrainer(model_name=args.reasoner_model, device=args.device)
    critic = BaseCritic(args.critic_model)
    
    # Train model
    history = reasoner_trainer.train_with_critic_feedback(
        dataset=dataset,
        critic=critic,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=os.path.join(args.output_dir, "reasoner_model")
    )
    
    print("Reasoner training completed")
    return history

def train_reasoner_dpo(args):
    print("Training Reasoner with DPO...")
    
    # Initialize trainer
    trainer = ReasonerTrainer(model_name=args.reasoner_model, device=args.device)
    
    # Prepare training data
    train_dataloader, val_dataloader = trainer.prepare_dpo_training_data(args.training_data)
    
    print(f"Training data: {len(train_dataloader.dataset)} examples")
    print(f"Validation data: {len(val_dataloader.dataset)} examples")
    
    # Train model
    history = trainer.train_dpo(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=os.path.join(args.output_dir, "reasoner_model_dpo")
    )
    
    print("Reasoner DPO training completed")
    return history

def evaluate_reasoner(args):
    print("Evaluating Reasoner...")
    
    # Load dataset
    dataset = load_aime_dataset()
    
    # Initialize evaluator
    evaluator = Evaluator(dataset)
    
    # Load model
    reasoner = BaseReasoner(args.reasoner_model)
    
    # Evaluate
    results = evaluator.evaluate_reasoner(
        reasoner=reasoner,
        output_path=os.path.join(args.output_dir, "reasoner_evaluation.json")
    )
    
    print(f"Evaluation completed with accuracy: {results['accuracy']:.4f}")
    return results

def evaluate_critic(args):
    print("Evaluating Critic...")
    
    # Load dataset
    dataset = load_aime_dataset()
    
    # Initialize evaluator
    evaluator = Evaluator(dataset)
    
    # Load model
    critic = BaseCritic(args.critic_model)
    
    # Evaluate
    results = evaluator.evaluate_critic(
        critic=critic,
        output_path=os.path.join(args.output_dir, "critic_evaluation.json")
    )
    
    print(f"Evaluation completed with average score: {results['average_score']:.4f}")
    return results

def inference(args):
    print("Running inference...")
    
    if args.problem is None:
        print("Error: Problem text is required for inference mode")
        return
    
    # Load models
    if args.use_critic:
        print("Using Critic-enhanced Reasoner")
        model = CriticEnhancedReasoner(
            reasoner_model_name=args.reasoner_model,
            critic_model_name=args.critic_model
        )
        
        # Solve problem with critic feedback
        result = model.solve(args.problem, max_iterations=3)
        
        print("\nProblem:")
        print(args.problem)
        print("\nFinal Solution:")
        print(result["final_solution"])
        
        # Show improvement history
        print("\nImprovement History:")
        for i, step in enumerate(result["history"]):
            print(f"\nIteration {i}:")
            if i > 0:
                print("Feedback:")
                print(step["feedback"])
            print("Solution:")
            print(step["solution"])
    else:
        print("Using Base Reasoner")
        model = BaseReasoner(args.reasoner_model)
        
        # Solve problem
        solutions = model.solve(args.problem)
        
        print("\nProblem:")
        print(args.problem)
        print("\nSolution:")
        print(solutions[0])
    
    return {"success": True}

def main():
    args = setup_argparse()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected mode
    if args.mode == "train_critic":
        train_critic(args)
    elif args.mode == "train_reasoner":
        train_reasoner_with_critic(args)
    elif args.mode == "train_reasoner_dpo":
        train_reasoner_dpo(args)
    elif args.mode == "evaluate_reasoner":
        evaluate_reasoner(args)
    elif args.mode == "evaluate_critic":
        evaluate_critic(args)
    elif args.mode == "inference":
        inference(args)
    else:
        print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main() 