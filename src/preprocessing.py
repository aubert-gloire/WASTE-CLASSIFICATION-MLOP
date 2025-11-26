"""
Data Preprocessing Module for Waste Classification
Handles image loading, augmentation, and dataset preparation
"""
import os
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, List
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    IMAGE_SIZE, CLASS_NAMES, BATCH_SIZE, 
    VALIDATION_SPLIT, AUGMENTATION_CONFIG
)


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image as numpy array
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]
    
    return img


def create_data_generators(train_dir: str, val_dir: str = None):
    """
    Create data generators for training and validation
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data (optional, use same as train_dir for split)
    
    Returns:
        Tuple of (train_generator, validation_generator)
    """
    # Check if we need to use validation_split (when val_dir is None or same as train_dir)
    use_validation_split = (val_dir is None) or (val_dir == train_dir)
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
        shear_range=AUGMENTATION_CONFIG['shear_range'],
        zoom_range=AUGMENTATION_CONFIG['zoom_range'],
        horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
        fill_mode=AUGMENTATION_CONFIG['fill_mode'],
        validation_split=VALIDATION_SPLIT if use_validation_split else 0.0
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT if use_validation_split else 0.0
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',  # Use sparse for integer labels (compatible with sparse_categorical_crossentropy)
        subset='training' if use_validation_split else None,
        shuffle=True
    )
    
    # Create validation generator
    if use_validation_split:
        # Use validation split from same directory
        validation_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='sparse',  # Use sparse for integer labels
            subset='validation',
            shuffle=False
        )
    else:
        # Use separate validation directory
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='sparse',  # Use sparse for integer labels
            shuffle=False
        )
    
    return train_generator, validation_generator


def organize_dataset(source_dir: str, output_dir: str, train_split: float = 0.8):
    """
    Organize dataset into train/validation splits
    
    Args:
        source_dir: Source directory with class subdirectories
        output_dir: Output directory for organized dataset
        train_split: Proportion of data for training
    """
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    val_dir = output_path / 'validation'
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Organizing dataset from {source_dir}...")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all images in this class
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        if len(images) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        # Split images
        train_images, val_images = train_test_split(
            images, 
            train_size=train_split, 
            random_state=42
        )
        
        # Create class directories
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        
        # Copy images
        for img in tqdm(train_images, desc=f"Copying {class_name} (train)"):
            shutil.copy2(img, train_dir / class_name / img.name)
        
        for img in tqdm(val_images, desc=f"Copying {class_name} (val)"):
            shutil.copy2(img, val_dir / class_name / img.name)
        
        print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"✓ Dataset organized in {output_dir}")
    return str(train_dir), str(val_dir)


def preprocess_uploaded_data(uploaded_dir: str, output_dir: str):
    """
    Preprocess newly uploaded data for retraining
    
    Args:
        uploaded_dir: Directory containing uploaded images
        output_dir: Output directory for processed images
    
    Returns:
        Number of processed images
    """
    uploaded_path = Path(uploaded_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for class_name in CLASS_NAMES:
        class_dir = uploaded_path / class_name
        if not class_dir.exists():
            continue
        
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(exist_ok=True)
        
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        for img_path in images:
            try:
                img = load_and_preprocess_image(str(img_path))
                # Save preprocessed image
                output_img_path = output_class_dir / img_path.name
                cv2.imwrite(str(output_img_path), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                processed_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"✓ Preprocessed {processed_count} uploaded images")
    return processed_count


def get_class_weights(train_generator):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        train_generator: Training data generator
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Get all labels
    labels = train_generator.classes
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return dict(enumerate(class_weights))


if __name__ == "__main__":
    print("Preprocessing module loaded successfully")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Classes: {CLASS_NAMES}")
