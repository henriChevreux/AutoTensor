"""
FashionMNIST Training Agent - An interactive AI agent for configuring and training models
"""

import argparse
import cmd
import json
import os
import shutil
import glob
import re
import numpy as np
from datetime import datetime
from gpt_api import get_gpt_reccomendation_from_tb

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Warning: TensorBoard not installed. Advanced analytics will be limited.")
    event_accumulator = None

from tensorvision import TensorVision

class TrainingAgent(cmd.Cmd):
    """Interactive command-line agent for configuring and training FashionMNIST models"""
    
    intro = """
    =================================================================
    🤖 FashionMNIST Training Agent 🤖
    =================================================================
    Welcome! I can help you configure and train FashionMNIST models.
    Type 'help' or '?' to list commands.
    Type 'train' to train with current settings.
    Type 'exit' to quit.
    """
    prompt = '(agent) '
    
    # Default hyperparameters
    default_hyperparams = {
        # Data parameters
        "data_dir": "./data",
        "batch_size": 128,
        "num_workers": 4,
        
        # Model parameters
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "dropout_rate": 0.2,
        
        # Training parameters
        "max_epochs": 1,
        "accelerator": "auto",
        "precision": "32-true",
        "checkpoint_dir": "checkpoints",
    }
    
    def __init__(self):
        super().__init__()
        self.hyperparams = self.default_hyperparams.copy()
        self.results = None
        self.pipeline = None
    
    def do_train(self, arg):
        """Train the model with current hyperparameters.
        
        Usage: train"""
        print("\nStarting training with these hyperparameters:")
        
        # Create and train the model
        self.pipeline = TensorVision()
        self.results = self.pipeline.train()
        
        # Print and save results
        print("\nTraining complete!")
        print(f"Best model path: {self.results['best_model_path']}")
        print(f"Best validation accuracy: {self.results['best_model_score']:.4f}")
    
    def do_exit(self, arg):
        """Exit the program.
        
        Usage: exit"""
        print("Thank you for using FashionMNIST Training Agent! Goodbye.")
        return True
    
    def do_quit(self, arg):
        """Exit the program.
        
        Usage: quit"""
        return self.do_exit(arg)
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def _analyze_tensorboard_logs(self, log_dir="tb_logs", experiment_dir="fashion_mnist"):
        """Analyze TensorBoard logs to extract training patterns and metrics.
        
        Args:
            log_dir: Path to TensorBoard logs.
            experiment_dir: Path to specific experiment directory. If None, analyze all.
            
        Returns:
            Dictionary of analyzed metrics and patterns
        """
        if event_accumulator is None:
            print("TensorBoard not installed. Cannot analyze logs.")
            return None
            
        # Find TensorBoard log directories
        version_dirs = glob.glob(f"{log_dir}/{experiment_dir}/*")
        print(version_dirs)
    
        if not version_dirs:
            print("No TensorBoard logs found.")
            return None
            
        results = get_gpt_reccomendation_from_tb(version_dirs)

        # print the GPT results
        print(results)
        
        return results
    
    def do_analysis(self, arg):
        """Analyze the current set of logs and get reccomendations.
        
        Usage: analysis"""
        self._analyze_model_performance()
    
    def _analyze_model_performance(self):
        """Analyze model performance across experiments and identify patterns.
        
        Returns:
            Dictionary with performance analysis and recommendations
        """
        
        # Analyze TensorBoard logs
        tb_analysis = self._analyze_tensorboard_logs()

def main():
    parser = argparse.ArgumentParser(description="FashionMNIST Training Agent")
    args = parser.parse_args()
    
    agent = TrainingAgent()
    agent.cmdloop()

if __name__ == "__main__":
    main() 