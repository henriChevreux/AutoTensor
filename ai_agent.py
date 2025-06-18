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
from gpt_api import get_gpt_reccomendation_from_tb, get_gpt_model_code_from_tb_with_current_model, save_model_code_to_file
from tensorvision import TensorVision
from prompts import get_intro_prompt, get_cmd_prompt
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Warning: TensorBoard not installed. Advanced analytics will be limited.")
    event_accumulator = None

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

        self.save_analysis_to_file(analysis_dir="analysis_results", version_dirs=version_dirs, results=results)
        
        return results
    
    def do_analysis(self, arg):
        """Analyze the current set of logs and get reccomendations.
        
        Usage: analysis"""
        self._analyze_model_performance()

    def do_generate_model(self, arg):
        """Generate a new model code based on TensorBoard analysis and save it as a Python file.
        
        If no filename is provided, defaults to 'generated_model.py'"""
        self._generate_model_code_from_tb()
    
    def _generate_model_code_from_tb(
            self, 
            output_filename = 'generated_model.py', 
            log_dir="tb_logs", 
            experiment_dir="fashion_mnist"):
        """Generate a new model based on TensorBoard analysis and save it as a Python file.
        If no filename is provided, defaults to 'generated_model.py'"""

        print(f"\nAnalyzing TensorBoard logs and generating improved model code...")

        # Get TensorBoard log directories
        log_dir = "tb_logs"
        experiment_dir = "fashion_mnist"
        version_dirs = glob.glob(f"{log_dir}/{experiment_dir}/*")

        if not version_dirs:
            print("No TensorBoard logs found. Cannot generate model without training data.")
            return

        # Generate model code from GPT
        try:
            model_code = get_gpt_model_code_from_tb_with_current_model(version_dirs)

            # Save the code to a file
            saved_file = save_model_code_to_file(model_code, output_filename)

            if saved_file:
                print(f"\n✅ Model code generated and saved to: {saved_file}")
                print("\nYou can now:")
                print(f"1. Review the code in {saved_file}")
                print("2. Import and use the model in your training pipeline")
                print("3. Run 'train' to test the new model")
            else:
                print("❌ Failed to save model code")

        except Exception as e:
            print(f"❌ Error generating model: {e}")
    
    def _analyze_model_performance(self):
        """Analyze model performance across experiments and identify patterns.
        
        Returns:
            Dictionary with performance analysis and recommendations
        """
        
        # Analyze TensorBoard logs
        tb_analysis = self._analyze_tensorboard_logs()
    
    def save_analysis_to_file(self, analysis_dir="analysis_results", version_dirs=None, results=None):
        """Save the analysis to a file"""

        # Create output directory for analysis results
        os.makedirs(analysis_dir, exist_ok=True)

        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = f"{analysis_dir}/tensorboard_analysis_{timestamp}.txt"

        try:
            # Save results to file with proper formatting
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("TENSORBOARD ANALYSIS RESULTS\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"TensorBoard Logs Analyzed: {len(version_dirs)} directories\n\n")

                f.write("Log Directories:\n")
                for i, dir_path in enumerate(version_dirs, 1):
                    f.write(f"  {i}. {dir_path}\n")
                f.write("\n")

                f.write("-" * 80 + "\n")
                f.write("GPT ANALYSIS AND RECOMMENDATIONS\n")
                f.write("-" * 80 + "\n\n")

                # Write the GPT results with proper formatting
                if results:
                    # Normalize line breaks to ensure proper formatting
                    formatted_results = results.replace('\r\n', '\n').replace('\r', '\n')
                    f.write(formatted_results)

                    # Ensure the file ends with a newline
                    if not formatted_results.endswith('\n'):
                        f.write('\n')
                else:
                    f.write("No analysis results available.\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF ANALYSIS\n")
                f.write("=" * 80 + "\n")

            print(f"\n✅ TensorBoard analysis saved to: {analysis_file}")
            print(f"📄 Analysis file contains {len(results) if results else 0} characters")

            # Also print the results to console
            print("\n" + "=" * 50)
            print("GPT ANALYSIS RESULTS")
            print("=" * 50)
            print(results)

        except Exception as e:
            print(f"❌ Error saving analysis to file: {e}")
            # Still print results to console even if file saving fails
            print("\n" + "=" * 50)
            print("GPT ANALYSIS RESULTS")
            print("=" * 50)
            print(results)

def main():
    parser = argparse.ArgumentParser(description="FashionMNIST Training Agent")
    args = parser.parse_args()
    
    agent = TrainingAgent()
    agent.cmdloop()

if __name__ == "__main__":
    main() 