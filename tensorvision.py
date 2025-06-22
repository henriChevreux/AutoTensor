"""
TensorVision: High-level API for FashionMNIST CNN training with comprehensive logging
"""
import os
from typing import Dict, Any, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from data import FashionMNISTDataModule
import importlib

class TensorVision:
    """High-level API for managing the FashionMNIST CNN training pipeline with comprehensive logging"""
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        dropout_rate: float = 0.2,
        max_epochs: int = 1,
        seed: int = 42,
        accelerator: str = "auto",
        devices: int = 1,
        precision: str = "32-true",
        checkpoint_dir: str = "checkpoints",
        model_module: str = "generated_model",
        model_class: str = "FashionMNISTCNN",
        **kwargs
    ):
        """
        Initialize the TensorVision pipeline with the specified parameters.
        
        Args:
            data_dir: Directory to store dataset
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            dropout_rate: Dropout rate for model layers
            max_epochs: Maximum number of epochs for training
            seed: Random seed for reproducibility
            accelerator: Hardware accelerator to use (auto, cpu, gpu, etc.)
            devices: Number of devices to use
            precision: Precision for training
            checkpoint_dir: Directory to save model checkpoints
            **kwargs: Additional arguments to pass to the trainer
        """
        # Import the model dynamically
        self.model_module_name = model_module
        self.model_class_name = model_class
        model_module = importlib.import_module(model_module)
        model_class = getattr(model_module, model_class)
        
        # Set random seed
        pl.seed_everything(seed)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save configuration
        self.config = {
            "data_dir": data_dir,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "max_epochs": max_epochs,
            "seed": seed,
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
            "checkpoint_dir": checkpoint_dir,
            **kwargs
        }
        
        # Initialize components
        self.data_module = FashionMNISTDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        self.model = model_class(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate
        )
        
        # Initialize tensorboard logger
        self.logger = TensorBoardLogger(
            save_dir="tb_logs",
            name="fashion_mnist",
            version=None,  # auto-incremented version
            log_graph=True,
            default_hp_metric=False  # don't log hp_metric which is just a placeholder
        )
        
        config={
            "architecture": "CNN",
            "dataset": "FashionMNIST",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": max_epochs,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
        }

        self.logger.log_hyperparams(config)
        
        # Define callbacks
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=checkpoint_dir,
            filename="fashion_mnist-{epoch:02d}-{val/acc:.4f}",
            save_top_k=1,
            mode="min",
        )
        
        self.early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min",
            verbose=True
        )
        
        self.callbacks = [
            self.checkpoint_callback,
            self.early_stop_callback,
            LearningRateMonitor(logging_interval="step")
        ]
        
        # Initialize trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=self.logger,
            callbacks=self.callbacks,
            log_every_n_steps=50,
            #precision=precision,
            deterministic=True,
            **kwargs
        )
    
    def train(self) -> Dict[str, Any]:
        """Train the model and return metrics"""
        print(f"Training model {self.model_class_name} from {self.model_module_name}")
        # Train model
        self.trainer.fit(self.model, self.data_module)
        
        # Test model
        test_results = self.trainer.test(self.model, self.data_module)
        
        # Get best model information
        results = {
            "best_model_path": self.checkpoint_callback.best_model_path,
            "best_model_score": self.checkpoint_callback.best_model_score,
            "test_results": test_results
        }
        
        return results
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint"""
        self.model = self.model.load_from_checkpoint(checkpoint_path)
    
    def predict(self, dataloader: Optional[Any] = None):
        """Run prediction on dataloader or test dataloader if None"""
        if dataloader is None:
            dataloader = self.data_module.test_dataloader()
        return self.trainer.predict(self.model, dataloaders=dataloader)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "TensorVision":
        """Create a new TensorVision instance with a model loaded from checkpoint"""
        instance = cls(**kwargs)
        instance.load_checkpoint(checkpoint_path)
        return instance 