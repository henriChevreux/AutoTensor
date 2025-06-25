"""
FashionMNIST Training Agent - An interactive AI agent for configuring and training models
"""

import argparse
import cmd
from prompts import get_intro_prompt, get_cmd_prompt
from helpers.generate_model_helpers import generate_model_pipeline
from helpers.analysis_helpers import analysis_pipeline
from helpers.train_helpers import train_pipeline, get_latest_model_module

class TrainingAgent(cmd.Cmd):
    """Interactive command-line agent for configuring and training FashionMNIST models"""
    
    intro = get_intro_prompt()
    prompt = get_cmd_prompt()
    
    def __init__(self):
        super().__init__()
        self.results = None
        self.models_dir = "models"
        self.experiment_name = "fashion_mnist"
        self.model_module = "models.fashion_mnist.model_1"
        self.model_class = "FashionMNISTCNN"
        self.best_model_checkpoint = None
    
    def do_train(self, arg):
        """Train the model with current hyperparameters.
        
        Usage: train"""
        latest_model_module = get_latest_model_module(self.models_dir, self.experiment_name)
        print(f"Latest model found: {latest_model_module}")
        best_model_checkpoint = train_pipeline(latest_model_module, self.model_class)
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
    
    def do_auto_learn(self, arg):
        """Loop n times through the following steps:
        1. Train the model
        2. Analyze the model
        3. Generate a new model
        Usage: auto_learn <number of times to loop>"""
        n = int(arg)
        print(f"Auto learning {n} times...")
        print("--------------------------------")
        for i in range(n):
            print(f"Auto learning iteration {i+1} of {n}...")
            self.do_train("")
            self.do_analysis("")
            self.do_generate_model("") 
            print("--------------------------------")
        print("Auto learning complete!")


def main():
    parser = argparse.ArgumentParser(description="FashionMNIST Training Agent")
    args = parser.parse_args()
    
    agent = TrainingAgent()
    agent.cmdloop()

if __name__ == "__main__":
    main() 