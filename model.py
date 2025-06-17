import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class FashionMNISTCNN(pl.LightningModule):
    def __init__(self, learning_rate=0.001, weight_decay=1e-5, dropout_rate=0.2):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Simplified CNN architecture for grayscale images
        self.features = nn.Sequential(
            # Input is 1 channel (grayscale)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(dropout_rate),
        )
        
        # FashionMNIST is 28x28, after 3 max-pooling layers (stride 2 each), we get 28/(2^3) = 3.5 -> 3
        # Calculating final feature size: 128 channels × 3 × 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10),
        )
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        
        self.val_precision = Precision(task="multiclass", num_classes=10, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=10, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=10, average="macro")
        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=10)
        
        # For per-class metrics
        self.val_per_class_acc = Accuracy(task="multiclass", num_classes=10, average=None)
        
        # Store example predictions for visualization
        self.example_input_array = torch.rand(1, 1, 28, 28)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Logging
        self.train_acc(preds, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.val_per_class_acc(preds, y)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/precision", self.val_precision)
        self.log("val/recall", self.val_recall)
        self.log("val/f1", self.val_f1)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        self.test_acc(preds, y)
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }