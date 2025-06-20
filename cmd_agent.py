"""
FashionMNIST Training Agent - An interactive AI agent for configuring and training models
"""

import argparse
import cmd
from prompts import get_intro_prompt, get_cmd_prompt
from helpers.generate_model_helpers import generate_model_pipeline
from helpers.analysis_helpers import analysis_pipeline
from helpers.train_helpers import train_pipeline

class TrainingAgent(cmd.Cmd):
    """Interactive command-line agent for configuring and training FashionMNIST models"""
    
    intro = get_intro_prompt()
    prompt = get_cmd_prompt()
    
    def __init__(self):
        super().__init__()
        self.results = None
        self.model_module = "models.fashion_mnist.model_1"
        self.model_class = "FashionMNISTCNN"
        self.best_model_checkpoint = None
    
    def do_train(self, arg):
        """Train the model with current hyperparameters.
        
        Usage: train"""
        best_model_checkpoint = train_pipeline(self.model_module, self.model_class)
        self.best_model_checkpoint = best_model_checkpoint
    
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