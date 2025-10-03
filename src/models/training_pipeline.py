# -*- coding: utf-8 -*-
"""
Training pipeline for detection and recognition models.
"""

import os
import json
import logging
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str
    model_type: str  # 'detection' or 'recognition'
    architecture: str
    learning_rate: float = 0.001
    batch_size: int = 16
    num_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_interval: int = 5
    data_augmentation: bool = True
    mixed_precision: bool = False
    gradient_clipping: float = 1.0
    weight_decay: float = 0.0001
    scheduler_type: str = "cosine"  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring progress."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """Deserialize from dictionary."""
        return cls(**data)


class DataAugmentation:
    """Data augmentation utilities for training."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize data augmentation."""
        self.config = config
        self.enabled = config.data_augmentation
    
    def augment_detection_data(self, image: np.ndarray, 
                             bboxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """Apply data augmentation for detection training."""
        if not self.enabled:
            return image, bboxes
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image)
            # Adjust bounding boxes for horizontal flip
            h, w = image.shape[:2]
            bboxes = [(w - x2, y1, w - x1, y2) for x1, y1, x2, y2 in bboxes]
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Random noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 5, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image, bboxes
    
    def augment_recognition_data(self, image: np.ndarray, text: str) -> Tuple[np.ndarray, str]:
        """Apply data augmentation for recognition training."""
        if not self.enabled:
            return image, text
        
        # Random rotation (small angles)
        if random.random() > 0.7:
            angle = random.uniform(-5, 5)
            # Apply rotation (simplified - would need proper implementation)
            pass
        
        # Random blur
        if random.random() > 0.8:
            # Apply Gaussian blur (simplified - would need proper implementation)
            pass
        
        # Random brightness/contrast (same as detection)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        return image, text


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop early."""
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class DetectionTrainingPipeline:
    """Training pipeline for text detection models."""
    
    def __init__(self, config: TrainingConfig, output_dir: str):
        """Initialize detection training pipeline."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_augmentation = DataAugmentation(config)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Metrics tracking
        self.training_history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Tensorboard logging
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        logger.info(f"Detection training pipeline initialized with device: {self.device}")
    
    def setup_model(self, model_architecture: str, **model_kwargs) -> nn.Module:
        """Setup the detection model."""
        if model_architecture == "YOLO":
            # Simplified YOLO model setup
            model = self._create_yolo_model(**model_kwargs)
        elif model_architecture == "RCNN":
            # Simplified R-CNN model setup
            model = self._create_rcnn_model(**model_kwargs)
        else:
            raise ValueError(f"Unsupported detection architecture: {model_architecture}")
        
        model = model.to(self.device)
        self.model = model
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        if self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5
            )
        
        logger.info(f"Model setup completed: {model_architecture}")
        return model
    
    def _create_yolo_model(self, num_classes: int = 1, input_size: Tuple[int, int] = (640, 640)) -> nn.Module:
        """Create a simplified YOLO model for demonstration."""
        # This is a simplified version - in practice, you'd use a proper YOLO implementation
        class SimpleYOLO(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.head = nn.Linear(256, num_classes * 5)  # 4 bbox coords + 1 confidence
            
            def forward(self, x):
                features = self.backbone(x)
                features = features.view(features.size(0), -1)
                output = self.head(features)
                return output
        
        return SimpleYOLO(num_classes)
    
    def _create_rcnn_model(self, num_classes: int = 1) -> nn.Module:
        """Create a simplified R-CNN model for demonstration."""
        # This is a placeholder - in practice, you'd use torchvision's implementation
        class SimpleRCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128 * 7 * 7, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                features = self.backbone(x)
                features = features.view(features.size(0), -1)
                output = self.classifier(features)
                return output
        
        return SimpleRCNN(num_classes)
    
    def setup_data_loaders(self, train_dataset: Dataset, val_dataset: Dataset):
        """Setup training and validation data loaders."""
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Data loaders setup: train={len(train_dataset)}, val={len(val_dataset)}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss (simplified - would need proper detection loss)
            loss = nn.MSELoss()(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        
        return avg_loss, epoch_time
    
    def validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = nn.MSELoss()(outputs, targets)
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict(),
            'training_history': [m.to_dict() for m in self.training_history],
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.best_model_path = str(best_path)
            logger.info(f"New best model saved at epoch {epoch}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore training history
        history_data = checkpoint.get('training_history', [])
        self.training_history = [TrainingMetrics.from_dict(m) for m in history_data]
        
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              resume_from: Optional[str] = None) -> str:
        """Run the complete training pipeline."""
        logger.info("Starting detection model training")
        
        # Setup data loaders
        self.setup_data_loaders(train_dataset, val_dataset)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss, epoch_time = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            self.training_history.append(metrics)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save final model
        self.save_checkpoint(epoch, False)
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            history_data = [m.to_dict() for m in self.training_history]
            json.dump(history_data, f, indent=2)
        
        self.writer.close()
        
        logger.info("Training completed")
        return self.best_model_path or str(self.output_dir / f"checkpoint_epoch_{epoch}.pth")


print("Training pipeline module loaded successfully")

class
 RecognitionTrainingPipeline:
    """Training pipeline for text recognition models."""
    
    def __init__(self, config: TrainingConfig, output_dir: str):
        """Initialize recognition training pipeline."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_augmentation = DataAugmentation(config)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Metrics tracking
        self.training_history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Tensorboard logging
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        logger.info(f"Recognition training pipeline initialized with device: {self.device}")
    
    def setup_model(self, model_architecture: str, vocab_size: int, **model_kwargs) -> nn.Module:
        """Setup the recognition model."""
        if model_architecture == "CRNN":
            model = self._create_crnn_model(vocab_size, **model_kwargs)
        elif model_architecture == "TrOCR":
            model = self._create_trocr_model(vocab_size, **model_kwargs)
        else:
            raise ValueError(f"Unsupported recognition architecture: {model_architecture}")
        
        model = model.to(self.device)
        self.model = model
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        if self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5
            )
        
        logger.info(f"Model setup completed: {model_architecture}")
        return model
    
    def _create_crnn_model(self, vocab_size: int, hidden_size: int = 256) -> nn.Module:
        """Create a simplified CRNN model for demonstration."""
        class SimpleCRNN(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                # CNN backbone
                self.cnn = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, None))  # Keep width dimension
                )
                
                # RNN layers
                self.rnn = nn.LSTM(256, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
                
                # Output layer
                self.classifier = nn.Linear(hidden_size * 2, vocab_size)
            
            def forward(self, x):
                # CNN feature extraction
                features = self.cnn(x)  # [B, 256, 1, W]
                features = features.squeeze(2)  # [B, 256, W]
                features = features.permute(0, 2, 1)  # [B, W, 256]
                
                # RNN sequence modeling
                rnn_out, _ = self.rnn(features)  # [B, W, hidden_size*2]
                
                # Classification
                output = self.classifier(rnn_out)  # [B, W, vocab_size]
                
                return output
        
        return SimpleCRNN(vocab_size, hidden_size)
    
    def _create_trocr_model(self, vocab_size: int) -> nn.Module:
        """Create a simplified TrOCR model for demonstration."""
        class SimpleTrOCR(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                # Vision encoder (simplified)
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 32))
                )
                
                # Text decoder (simplified transformer)
                self.text_decoder = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(d_model=512, nhead=8),
                    num_layers=6
                )
                
                self.embedding = nn.Embedding(vocab_size, 512)
                self.output_projection = nn.Linear(512, vocab_size)
            
            def forward(self, images, text_tokens=None):
                # Vision encoding
                visual_features = self.vision_encoder(images)  # [B, 128, 8, 32]
                visual_features = visual_features.flatten(2).permute(2, 0, 1)  # [256, B, 128]
                
                if text_tokens is not None:
                    # Training mode
                    text_embeddings = self.embedding(text_tokens)  # [B, seq_len, 512]
                    text_embeddings = text_embeddings.permute(1, 0, 2)  # [seq_len, B, 512]
                    
                    decoder_output = self.text_decoder(text_embeddings, visual_features)
                    output = self.output_projection(decoder_output)
                    return output.permute(1, 0, 2)  # [B, seq_len, vocab_size]
                else:
                    # Inference mode (simplified)
                    return visual_features
        
        return SimpleTrOCR(vocab_size)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (images, text_tokens, text_lengths) in enumerate(self.train_loader):
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, text_tokens[:, :-1])  # Exclude last token for input
            
            # Calculate loss (cross-entropy for text recognition)
            targets = text_tokens[:, 1:]  # Exclude first token for targets
            loss = nn.CrossEntropyLoss(ignore_index=0)(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        
        return avg_loss, epoch_time
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_chars = 0
        total_chars = 0
        
        with torch.no_grad():
            for images, text_tokens, text_lengths in self.val_loader:
                images = images.to(self.device)
                text_tokens = text_tokens.to(self.device)
                
                outputs = self.model(images, text_tokens[:, :-1])
                targets = text_tokens[:, 1:]
                
                # Calculate loss
                loss = nn.CrossEntropyLoss(ignore_index=0)(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                # Calculate character accuracy
                predictions = torch.argmax(outputs, dim=-1)
                mask = targets != 0  # Ignore padding tokens
                correct_chars += (predictions == targets)[mask].sum().item()
                total_chars += mask.sum().item()
        
        avg_loss = total_loss / total_samples
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        
        return avg_loss, char_accuracy
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              resume_from: Optional[str] = None) -> str:
        """Run the complete training pipeline."""
        logger.info("Starting recognition model training")
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss, epoch_time = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_accuracy = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            self.training_history.append(metrics)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, LR: {current_lr:.6f}")
        
        # Save final model
        self.save_checkpoint(epoch, False)
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            history_data = [m.to_dict() for m in self.training_history]
            json.dump(history_data, f, indent=2)
        
        self.writer.close()
        
        logger.info("Training completed")
        return self.best_model_path or str(self.output_dir / f"checkpoint_epoch_{epoch}.pth")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict(),
            'training_history': [m.to_dict() for m in self.training_history],
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.best_model_path = str(best_path)
            logger.info(f"New best model saved at epoch {epoch}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore training history
        history_data = checkpoint.get('training_history', [])
        self.training_history = [TrainingMetrics.from_dict(m) for m in history_data]
        
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch


class TrainingManager:
    """High-level manager for training pipelines."""
    
    def __init__(self, base_output_dir: str = "training_outputs"):
        """Initialize training manager."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_trainings: Dict[str, Any] = {}
        
        logger.info(f"Training manager initialized with output dir: {self.base_output_dir}")
    
    def create_detection_pipeline(self, config: TrainingConfig) -> DetectionTrainingPipeline:
        """Create a detection training pipeline."""
        output_dir = self.base_output_dir / f"detection_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pipeline = DetectionTrainingPipeline(config, str(output_dir))
        
        self.active_trainings[f"detection_{config.model_name}"] = pipeline
        
        logger.info(f"Created detection training pipeline for {config.model_name}")
        return pipeline
    
    def create_recognition_pipeline(self, config: TrainingConfig) -> RecognitionTrainingPipeline:
        """Create a recognition training pipeline."""
        output_dir = self.base_output_dir / f"recognition_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pipeline = RecognitionTrainingPipeline(config, str(output_dir))
        
        self.active_trainings[f"recognition_{config.model_name}"] = pipeline
        
        logger.info(f"Created recognition training pipeline for {config.model_name}")
        return pipeline
    
    def get_training_status(self, model_name: str, model_type: str) -> Dict[str, Any]:
        """Get status of a training pipeline."""
        key = f"{model_type}_{model_name}"
        if key not in self.active_trainings:
            return {"status": "not_found"}
        
        pipeline = self.active_trainings[key]
        
        if not pipeline.training_history:
            return {"status": "not_started"}
        
        latest_metrics = pipeline.training_history[-1]
        
        return {
            "status": "training",
            "current_epoch": latest_metrics.epoch,
            "total_epochs": pipeline.config.num_epochs,
            "train_loss": latest_metrics.train_loss,
            "val_loss": latest_metrics.val_loss,
            "best_val_loss": pipeline.best_val_loss,
            "learning_rate": latest_metrics.learning_rate
        }
    
    def stop_training(self, model_name: str, model_type: str):
        """Stop a training pipeline."""
        key = f"{model_type}_{model_name}"
        if key in self.active_trainings:
            # In a real implementation, you'd need to handle stopping gracefully
            del self.active_trainings[key]
            logger.info(f"Stopped training for {model_name}")
    
    def list_active_trainings(self) -> List[str]:
        """List all active training pipelines."""
        return list(self.active_trainings.keys())


print("Training pipeline module completed successfully")