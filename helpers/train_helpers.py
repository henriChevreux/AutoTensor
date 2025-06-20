from tensorvision import TensorVision

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