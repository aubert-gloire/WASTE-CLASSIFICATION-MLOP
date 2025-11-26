"""
Model Creation and Training Module
Builds MobileNetV2-based waste classification model
"""
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, 
    ReduceLROnPlateau, TensorBoard
)
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    NUM_CLASSES, IMAGE_SIZE, LEARNING_RATE,
    EPOCHS, MODEL_PATH, MODEL_HISTORY_PATH, LOG_DIR
)


def create_model(input_shape: Tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3),
                 num_classes: int = NUM_CLASSES,
                 trainable_base: bool = False) -> keras.Model:
    """
    Create MobileNetV2-based classification model
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        trainable_base: Whether to make base model trainable
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = trainable_base
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation layer (applied during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model


def get_callbacks(model_path: str = str(MODEL_PATH)) -> list:
    """
    Create training callbacks
    
    Args:
        model_path: Path to save the best model
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=str(LOG_DIR / f"fit_{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            histogram_freq=1
        )
    ]
    
    return callbacks


def train_model(train_generator, 
                validation_generator,
                epochs: int = EPOCHS,
                model_path: str = str(MODEL_PATH),
                class_weights: Dict = None) -> Tuple[keras.Model, dict]:
    """
    Train the waste classification model
    
    Args:
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of training epochs
        model_path: Path to save the model
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        Tuple of (trained_model, history_dict)
    """
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    # Create model
    model = create_model()
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {train_generator.batch_size}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Steps per epoch: {len(train_generator)}")
    print("="*60 + "\n")
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=get_callbacks(model_path),
        class_weight=class_weights,
        verbose=1
    )
    
    # Save training history
    with open(MODEL_HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ History saved to: {MODEL_HISTORY_PATH}")
    
    return model, history.history


def fine_tune_model(model: keras.Model,
                    train_generator,
                    validation_generator,
                    epochs: int = 10,
                    initial_epoch: int = 0) -> Tuple[keras.Model, dict]:
    """
    Fine-tune the model by unfreezing base layers
    
    Args:
        model: Pre-trained model
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of fine-tuning epochs
        initial_epoch: Starting epoch number
    
    Returns:
        Tuple of (fine_tuned_model, history_dict)
    """
    print("\n" + "="*60)
    print("FINE-TUNING MODEL")
    print("="*60)
    
    # Unfreeze base model layers
    base_model = model.layers[3]  # Assuming MobileNetV2 is the 4th layer
    base_model.trainable = True
    
    # Freeze early layers, unfreeze last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    print(f"Trainable layers: {len([l for l in model.layers if l.trainable])}")
    
    # Continue training
    history = model.fit(
        train_generator,
        epochs=initial_epoch + epochs,
        initial_epoch=initial_epoch,
        validation_data=validation_generator,
        callbacks=get_callbacks(str(MODEL_PATH).replace('.h5', '_finetuned.h5')),
        verbose=1
    )
    
    print("\n✓ Fine-tuning complete")
    
    return model, history.history


def load_trained_model(model_path: str = str(MODEL_PATH)) -> keras.Model:
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    return model


def get_model_info(model: keras.Model) -> dict:
    """
    Get model information and statistics
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with model information
    """
    try:
        trainable = sum([int(tf.size(w)) for w in model.trainable_weights])
        non_trainable = sum([int(tf.size(w)) for w in model.non_trainable_weights])
    except:
        # Fallback if eager execution not available
        trainable = 0
        non_trainable = 0
    
    return {
        'total_params': model.count_params(),
        'trainable_params': trainable,
        'non_trainable_params': non_trainable,
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }


if __name__ == "__main__":
    print("Model module loaded successfully")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
