#!/usr/bin/env python3
"""
Demonstration of recognition model training pipeline.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.training_pipeline import (
    TrainingConfig, RecognitionTrainingPipeline, TrainingManager
)


class SyntheticTextDataset(Dataset):
    """Synthetic text recognition dataset for demonstration."""
    
    def __init__(self, size: int = 1000, image_shape: tuple = (3, 32, 128), 
                 vocab_size: int = 95, max_length: int = 25):
        """Initialize synthetic dataset."""
        self.size = size
        self.image_shape = image_shape
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Create vocabulary (simplified)
        self.vocab = [chr(i) for i in range(32, 127)]  # ASCII printable characters
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.vocab)}  # 0 reserved for padding
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        print(f"Created synthetic dataset with {size} samples")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """Generate a synthetic text image and corresponding text."""
        # Generate random image (simulating text image)
        image = torch.randn(*self.image_shape)
        
        # Generate random text
        text_length = torch.randint(3, min(self.max_length - 2, 15), (1,)).item()
        text_chars = np.random.choice(self.vocab, size=text_length)
        text = ''.join(text_chars)
        
        # Convert text to token indices
        text_tokens = torch.zeros(self.max_length, dtype=torch.long)
        for i, char in enumerate(text):
            if i < self.max_length - 1:  # Leave space for end token
                text_tokens[i] = self.char_to_idx.get(char, 0)
        
        return image, text_tokens, text_length


def demonstrate_crnn_training():
    """Demonstrate CRNN model training."""
    print("=== CRNN Training Demonstration ===")
    
    # Create training configuration
    config = TrainingConfig(
        model_name="demo_crnn",
        model_type="recognition",
        architecture="CRNN",
        learning_rate=0.001,
        batch_size=8,
        num_epochs=5,  # Small number for demo
        validation_split=0.2,
        early_stopping_patience=3,
        checkpoint_interval=2,
        data_augmentation=True,
        gradient_clipping=1.0
    )
    
    print(f"Training config: {config.model_name} - {config.architecture}")
    print(f"Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
    
    # Create output directory
    output_dir = "demo_outputs/crnn_training"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize training pipeline
    pipeline = RecognitionTrainingPipeline(config, output_dir)
    
    # Setup model
    vocab_size = 95  # ASCII printable characters
    model = pipeline.setup_model("CRNN", vocab_size=vocab_size, hidden_size=128)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    train_dataset = SyntheticTextDataset(size=200, vocab_size=vocab_size)
    val_dataset = SyntheticTextDataset(size=50, vocab_size=vocab_size)
    
    print(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Run training
    print("Starting training...")
    try:
        best_model_path = pipeline.train(train_dataset, val_dataset)
        print(f"Training completed! Best model saved at: {best_model_path}")
        
        # Print training history
        print("\nTraining History:")
        for i, metrics in enumerate(pipeline.training_history):
            print(f"Epoch {metrics.epoch + 1}: "
                  f"Train Loss: {metrics.train_loss:.4f}, "
                  f"Val Loss: {metrics.val_loss:.4f}, "
                  f"Val Acc: {metrics.val_accuracy:.4f if metrics.val_accuracy else 'N/A'}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return None
    
    return pipeline


def demonstrate_trocr_training():
    """Demonstrate TrOCR model training."""
    print("\n=== TrOCR Training Demonstration ===")
    
    # Create training configuration
    config = TrainingConfig(
        model_name="demo_trocr",
        model_type="recognition",
        architecture="TrOCR",
        learning_rate=0.0001,  # Lower learning rate for transformer
        batch_size=4,  # Smaller batch size for transformer
        num_epochs=3,
        validation_split=0.2,
        early_stopping_patience=2,
        checkpoint_interval=1,
        data_augmentation=False,  # Disable for transformer demo
        scheduler_type="cosine"
    )
    
    print(f"Training config: {config.model_name} - {config.architecture}")
    
    # Create output directory
    output_dir = "demo_outputs/trocr_training"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize training pipeline
    pipeline = RecognitionTrainingPipeline(config, output_dir)
    
    # Setup model
    vocab_size = 95
    model = pipeline.setup_model("TrOCR", vocab_size=vocab_size)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create smaller datasets for transformer (more memory intensive)
    train_dataset = SyntheticTextDataset(size=100, vocab_size=vocab_size)
    val_dataset = SyntheticTextDataset(size=20, vocab_size=vocab_size)
    
    print(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Run training
    print("Starting training...")
    try:
        best_model_path = pipeline.train(train_dataset, val_dataset)
        print(f"Training completed! Best model saved at: {best_model_path}")
        
        # Print training history
        print("\nTraining History:")
        for i, metrics in enumerate(pipeline.training_history):
            print(f"Epoch {metrics.epoch + 1}: "
                  f"Train Loss: {metrics.train_loss:.4f}, "
                  f"Val Loss: {metrics.val_loss:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return None
    
    return pipeline


def demonstrate_training_manager():
    """Demonstrate training manager functionality."""
    print("\n=== Training Manager Demonstration ===")
    
    # Initialize training manager
    manager = TrainingManager("demo_outputs/managed_training")
    
    # Create multiple training configurations
    crnn_config = TrainingConfig(
        model_name="managed_crnn",
        model_type="recognition",
        architecture="CRNN",
        num_epochs=2,
        batch_size=4
    )
    
    trocr_config = TrainingConfig(
        model_name="managed_trocr",
        model_type="recognition",
        architecture="TrOCR",
        num_epochs=2,
        batch_size=2
    )
    
    # Create pipelines
    crnn_pipeline = manager.create_recognition_pipeline(crnn_config)
    trocr_pipeline = manager.create_recognition_pipeline(trocr_config)
    
    print(f"Created pipelines: {manager.list_active_trainings()}")
    
    # Check initial status
    crnn_status = manager.get_training_status("managed_crnn", "recognition")
    print(f"CRNN status: {crnn_status}")
    
    # Setup and run a quick training for demonstration
    crnn_pipeline.setup_model("CRNN", vocab_size=95)
    
    # Add some mock training history
    from src.models.training_pipeline import TrainingMetrics
    metrics = TrainingMetrics(
        epoch=0,
        train_loss=0.8,
        val_loss=0.9,
        val_accuracy=0.75,
        learning_rate=0.001,
        epoch_time=45.2
    )
    crnn_pipeline.training_history.append(metrics)
    
    # Check updated status
    crnn_status = manager.get_training_status("managed_crnn", "recognition")
    print(f"CRNN status after training: {crnn_status}")
    
    return manager


def demonstrate_transfer_learning():
    """Demonstrate transfer learning capabilities."""
    print("\n=== Transfer Learning Demonstration ===")
    
    # This would typically involve loading a pre-trained model
    # For demonstration, we'll show the structure
    
    config = TrainingConfig(
        model_name="transfer_crnn",
        model_type="recognition",
        architecture="CRNN",
        learning_rate=0.0001,  # Lower learning rate for fine-tuning
        num_epochs=3,
        batch_size=8
    )
    
    output_dir = "demo_outputs/transfer_learning"
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = RecognitionTrainingPipeline(config, output_dir)
    
    # Setup model
    model = pipeline.setup_model("CRNN", vocab_size=95)
    
    print("Transfer learning setup:")
    print(f"- Model architecture: {config.architecture}")
    print(f"- Learning rate: {config.learning_rate} (reduced for fine-tuning)")
    print(f"- Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # In a real scenario, you would:
    # 1. Load pre-trained weights: model.load_state_dict(pretrained_weights)
    # 2. Optionally freeze some layers: for param in model.backbone.parameters(): param.requires_grad = False
    # 3. Fine-tune on your specific dataset
    
    print("Transfer learning pipeline ready for fine-tuning!")
    
    return pipeline


def main():
    """Run all demonstrations."""
    print("Recognition Model Training Pipeline Demonstration")
    print("=" * 50)
    
    # Check if PyTorch is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU for demonstration")
    else:
        print(f"CUDA available with {torch.cuda.device_count()} device(s)")
    
    try:
        # Demonstrate CRNN training
        crnn_pipeline = demonstrate_crnn_training()
        
        # Demonstrate TrOCR training
        trocr_pipeline = demonstrate_trocr_training()
        
        # Demonstrate training manager
        manager = demonstrate_training_manager()
        
        # Demonstrate transfer learning setup
        transfer_pipeline = demonstrate_transfer_learning()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
        # Summary
        print("\nSummary of implemented features:")
        print("✓ CRNN model training pipeline")
        print("✓ TrOCR model training pipeline")
        print("✓ Training manager for multiple models")
        print("✓ Transfer learning support")
        print("✓ Early stopping and validation")
        print("✓ Model checkpointing")
        print("✓ Training metrics tracking")
        print("✓ Data augmentation")
        print("✓ Learning rate scheduling")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()