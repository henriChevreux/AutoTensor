"""
FashionMNIST Training Agent - An interactive AI agent for configuring and training models
"""

import argparse
import cmd
import glob
import re
import numpy as np

from tensorvision import TensorVision
from prompts import get_intro_prompt, get_cmd_prompt
from helpers.generate_model_helpers import generate_model_pipeline
from helpers.analysis_helpers import analysis_pipeline

class TrainingAgent(cmd.Cmd):
    """Interactive command-line agent for configuring and training FashionMNIST models"""
    
    intro = get_intro_prompt()
    prompt = get_cmd_prompt()
    
    def __init__(self):
        super().__init__()
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
    
    def do_analysis(self, arg):
        """Analyze the current set of logs and get reccomendations.
        
        Usage: analysis"""
        analysis_pipeline()

    def do_generate_model(self, arg):
        """Generate a new model code based on TensorBoard analysis and save it as a Python file.
        
        If no filename is provided, defaults to 'generated_model.py'"""
        generate_model_pipeline()

def main():
    parser = argparse.ArgumentParser(description="FashionMNIST Training Agent")
    args = parser.parse_args()
    
    agent = TrainingAgent()
    agent.cmdloop()

if __name__ == "__main__":
    main() 