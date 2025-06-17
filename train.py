"""
Train the FashionMNIST model using the TensorVision pipeline
"""

from tensorvision import TensorVision

def main():
    # Create and train the model
    pipeline = TensorVision(
        # Data parameters
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        
        # Model parameters
        learning_rate=0.001,
        weight_decay=1e-5,
        dropout_rate=0.3,
        
        # Training parameters
        max_epochs=1,
        accelerator="auto",  # Use GPU if available
        precision="32-true",
    )
    
    # Train the model
    results = pipeline.train()
    
    # Print results
    print("Training complete!")
    print(f"Best model path: {results['best_model_path']}")
    print(f"Best validation accuracy: {results['best_model_score']:.4f}")
    print(f"Test results: {results['test_results']}")
    
    return results

if __name__ == "__main__":
    main() 