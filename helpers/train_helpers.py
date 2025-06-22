from tensorvision import TensorVision
import os
from helpers.helpers import get_max_version
import re

def train_pipeline(model_module, model_class):
    """Train the model with the given model path and pipeline"""
    # Create and train the model
    pipeline = TensorVision(model_module=model_module, model_class=model_class)
    results = pipeline.train()
    
    # Print and save results
    print("\nTraining complete!")
    print(f"Best model path: {results['best_model_path']}")
    print(f"Best validation accuracy: {results['best_model_score']:.4f}")
    return results['best_model_path']

def get_latest_model_module(models_dir, experiment_name):
    models_path = f"{models_dir}/{experiment_name}"
    max_version = get_max_version(models_path, "model", "py")
    return f"{models_dir}.{experiment_name}.model_{max_version}"