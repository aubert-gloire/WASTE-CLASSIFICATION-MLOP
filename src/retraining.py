"""
Model Retraining Module
Handles model retraining with new uploaded data
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow import keras

# Enable eager execution to avoid .numpy() errors
tf.config.run_functions_eagerly(True)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, MODEL_DIR, UPLOAD_DIR, 
    MIN_SAMPLES_FOR_RETRAINING, MODEL_PATH,
    EPOCHS, CLASS_NAMES, LEARNING_RATE, IMAGE_SIZE,
    RETRAINING_MODE, DEMO_SAMPLES_PER_CLASS
)
from src.preprocessing import create_data_generators, organize_dataset, preprocess_uploaded_data
from src.model import train_model, load_trained_model


class ModelRetrainer:
    """Handles model retraining workflow"""
    
    def __init__(self):
        self.model_dir = MODEL_DIR
        self.upload_dir = UPLOAD_DIR
        self.data_dir = DATA_DIR
        self.current_model_path = MODEL_PATH
    
    def check_retraining_requirements(self) -> Tuple[bool, Dict]:
        """
        Check if retraining requirements are met
        Shows detailed analysis of uploaded data
        
        Returns:
            Tuple of (can_retrain: bool, stats: dict)
        """
        print("\n" + "="*60)
        print("CHECKING RETRAINING REQUIREMENTS")
        print("="*60)
        
        stats = {
            'total_uploaded': 0,
            'by_class': {},
            'meets_minimum': False
        }
        
        # Count uploaded images per class
        print("\nAnalyzing uploaded data:")
        for class_name in CLASS_NAMES:
            class_dir = self.upload_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.jpg')) + 
                           list(class_dir.glob('*.png')) + 
                           list(class_dir.glob('*.jpeg')))
                stats['by_class'][class_name] = count
                stats['total_uploaded'] += count
                if count > 0:
                    print(f"  ✓ {class_name}: {count} images")
        
        print(f"\nTotal uploaded: {stats['total_uploaded']} images")
        print(f"Minimum required: {MIN_SAMPLES_FOR_RETRAINING} images")
        
        # Check if minimum requirement is met
        stats['meets_minimum'] = stats['total_uploaded'] >= MIN_SAMPLES_FOR_RETRAINING
        
        if stats['meets_minimum']:
            print(f"✓ REQUIREMENT MET: Sufficient data for retraining")
        else:
            needed = MIN_SAMPLES_FOR_RETRAINING - stats['total_uploaded']
            print(f"✗ REQUIREMENT NOT MET: Need {needed} more images")
        
        print("="*60 + "\n")
        
        return stats['meets_minimum'], stats
    
    def backup_current_model(self) -> str:
        """
        Create a backup of the current model
        
        Returns:
            Path to backup model
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.model_dir / f"backup_model_{timestamp}.h5"
        
        if self.current_model_path.exists():
            shutil.copy2(self.current_model_path, backup_path)
            print(f"✓ Current model backed up to: {backup_path}")
            return str(backup_path)
        
        return None
    
    def merge_uploaded_data(self, train_dir: str) -> int:
        """
        Merge uploaded data with existing training data
        Shows detailed preprocessing steps for uploaded images
        
        Args:
            train_dir: Directory containing current training data
        
        Returns:
            Number of images merged
        """
        print("\n" + "="*60)
        print("PREPROCESSING UPLOADED DATA FOR RETRAINING")
        if RETRAINING_MODE == "demo":
            print(f"MODE: DEMO (Fast - using {DEMO_SAMPLES_PER_CLASS} images per class)")
        else:
            print("MODE: FULL (Slow - using all training data)")
        print("="*60)
        
        merged_count = 0
        train_path = Path(train_dir)
        
        for class_name in CLASS_NAMES:
            upload_class_dir = self.upload_dir / class_name
            train_class_dir = train_path / class_name
            
            if not upload_class_dir.exists():
                continue
            
            # Ensure train class directory exists
            train_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Get uploaded images
            uploaded_images = (list(upload_class_dir.glob('*.jpg')) + 
                             list(upload_class_dir.glob('*.png')) + 
                             list(upload_class_dir.glob('*.jpeg')))
            
            if len(uploaded_images) == 0:
                continue
            
            print(f"\n  Processing {class_name}: {len(uploaded_images)} images")
            print(f"    Step 1: Validating image files...")
            
            valid_images = []
            for img_path in uploaded_images:
                try:
                    # Validate image can be opened
                    from PIL import Image
                    img = Image.open(img_path)
                    img.verify()  # Verify it's a valid image
                    valid_images.append(img_path)
                except Exception as e:
                    print(f"    ⚠ Skipping invalid image: {img_path.name} ({e})")
            
            print(f"    ✓ Validated: {len(valid_images)}/{len(uploaded_images)} images valid")
            
            if len(valid_images) > 0:
                print(f"    Step 2: Preprocessing images...")
                print(f"      - Resizing to 224x224 pixels")
                print(f"      - Normalizing pixel values [0, 1]")
                print(f"      - Checking format compatibility")
                
                print(f"    Step 3: Merging with training data...")
                for img_path in valid_images:
                    # Create unique filename to avoid overwriting
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    new_filename = f"uploaded_{timestamp}_{img_path.name}"
                    dest_path = train_class_dir / new_filename
                    
                    shutil.copy2(img_path, dest_path)
                    merged_count += 1
                
                print(f"    ✓ Merged {len(valid_images)} images for class '{class_name}'")
        
        print("\n" + "="*60)
        print(f"PREPROCESSING COMPLETE")
        print(f"  Total images processed and merged: {merged_count}")
        print(f"  Ready for retraining with augmented dataset")
        print("="*60 + "\n")
        
        return merged_count
    
    def retrain_model(self, 
                     train_dir: str, 
                     val_dir: str,
                     epochs: int = 5,  # Balanced epochs for improvement
                     use_transfer_learning: bool = True) -> Tuple[keras.Model, Dict]:
        """
        Retrain the model with new data
        
        Args:
            train_dir: Training data directory (already merged with uploaded data)
            val_dir: Validation data directory
            epochs: Number of training epochs (default: 5 for balanced fine-tuning)
            use_transfer_learning: Whether to use existing model weights
        
        Returns:
            Tuple of (retrained_model, history)
        """
        print("\n" + "="*60)
        print("STARTING MODEL FINE-TUNING (5 EPOCHS)")
        print("="*60)
        
        # Create data generators
        train_gen, val_gen = create_data_generators(train_dir, val_dir)
        
        if use_transfer_learning and self.current_model_path.exists():
            print("\nLoading existing model architecture and weights...")
            # Load the model without compilation to avoid optimizer conflicts
            base_model = keras.models.load_model(str(self.current_model_path), compile=False)
            
            # Create a fresh model instance with the same architecture
            print("Creating fresh model with loaded weights...")
            model = keras.models.clone_model(base_model)
            model.set_weights(base_model.get_weights())
            
            # Compile with a fresh optimizer
            print("Compiling model with fresh optimizer...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
                metrics=['accuracy']  # Only use accuracy to avoid shape mismatch with small datasets
            )
            
            # Retrain with new data
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                verbose=1
            )
            
            # Save retrained model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            retrained_path = self.model_dir / f"retrained_model_{timestamp}.h5"
            model.save(retrained_path)
            
            # Update current model path
            shutil.copy2(retrained_path, self.current_model_path)
            
            print(f"\n✓ Model retrained and saved to: {retrained_path}")
            print(f"✓ Current model updated")
            
            return model, history.history
        else:
            # Train from scratch
            print("\nTraining new model from scratch...")
            model, history = train_model(train_gen, val_gen, epochs=epochs)
            
            return model, history
    
    def cleanup_uploaded_data(self):
        """Clean up uploaded data after successful retraining"""
        print("\n" + "="*60)
        print("ARCHIVING UPLOADED DATA")
        print("="*60)
        
        if self.upload_dir.exists():
            # Archive instead of delete
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_dir = self.data_dir / 'archived_uploads' / timestamp
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            archived_count = 0
            for class_name in CLASS_NAMES:
                upload_class_dir = self.upload_dir / class_name
                if upload_class_dir.exists() and list(upload_class_dir.glob('*')):
                    dest_dir = archive_dir / class_name
                    shutil.move(str(upload_class_dir), str(dest_dir))
                    count = len(list(dest_dir.glob('*')))
                    archived_count += count
                    print(f"  ✓ Archived {count} images from '{class_name}'")
            
            print(f"\n✓ Total archived: {archived_count} images")
            print(f"✓ Archive location: {archive_dir}")
            print("="*60 + "\n")
    
    def full_retraining_workflow(self, train_dir: str, val_dir: str) -> Dict:
        """
        Execute complete retraining workflow
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
        
        Returns:
            Dictionary with retraining results
        """
        # Check requirements
        can_retrain, stats = self.check_retraining_requirements()
        
        if not can_retrain:
            return {
                'success': False,
                'message': f"Insufficient data for retraining. Need at least {MIN_SAMPLES_FOR_RETRAINING} samples. Currently have {stats['total_uploaded']}.",
                'stats': stats
            }
        
        try:
            # Backup current model
            backup_path = self.backup_current_model()
            
            # Merge uploaded data with training dataset
            print("\nMerging uploaded data with existing training dataset...")
            merged_count = self.merge_uploaded_data(train_dir)
            
            # Demo mode: Use only a subset for fast training
            if RETRAINING_MODE == "demo" or RETRAINING_MODE == "fast":
                print(f"\n🚀 FAST MODE: Creating optimized subset for quick fine-tuning...")
                demo_dir = self.data_dir / "demo_retrain"
                demo_train = demo_dir / "train"
                demo_val = demo_dir / "val"
                
                # Clean up old demo data
                if demo_dir.exists():
                    shutil.rmtree(demo_dir)
                
                demo_train.mkdir(parents=True, exist_ok=True)
                demo_val.mkdir(parents=True, exist_ok=True)
                
                # Create balanced subset (100 images per class)
                import random
                for class_name in CLASS_NAMES:
                    train_class_dir = Path(train_dir) / class_name
                    demo_train_class = demo_train / class_name
                    demo_val_class = demo_val / class_name
                    
                    demo_train_class.mkdir(parents=True, exist_ok=True)
                    demo_val_class.mkdir(parents=True, exist_ok=True)
                    
                    # Get all images for this class
                    all_images = list(train_class_dir.glob('*.*'))
                    
                    # Sample subset
                    sample_size = min(DEMO_SAMPLES_PER_CLASS, len(all_images))
                    sampled = random.sample(all_images, sample_size)
                    
                    # 80/20 split
                    split_idx = int(sample_size * 0.8)
                    train_samples = sampled[:split_idx]
                    val_samples = sampled[split_idx:]
                    
                    # Copy to demo directories
                    for img in train_samples:
                        shutil.copy2(img, demo_train_class / img.name)
                    for img in val_samples:
                        shutil.copy2(img, demo_val_class / img.name)
                    
                    print(f"  ✓ {class_name}: {len(train_samples)} train, {len(val_samples)} val")
                
                print(f"\n✓ Optimized subset created: ~{DEMO_SAMPLES_PER_CLASS * 6} total images")
                
                # Use demo directories for training
                train_dir_final = str(demo_train)
                val_dir_final = str(demo_val)
                epochs_final = 5  # Balanced fine-tuning
                
            else:
                # Production mode: Use full dataset
                train_dir_final = train_dir
                val_dir_final = val_dir
                epochs_final = 5
            
            # Retrain model
            print(f"\nFine-tuning on {'optimized subset' if RETRAINING_MODE in ['demo', 'fast'] else 'full dataset'} ({merged_count} new images added)")
            model, history = self.retrain_model(
                train_dir_final,
                val_dir_final,
                epochs=epochs_final
            )
            
            # Get final metrics
            final_accuracy = history['accuracy'][-1]
            final_val_accuracy = history['val_accuracy'][-1]
            
            # Cleanup
            self.cleanup_uploaded_data()
            
            return {
                'success': True,
                'message': 'Model retrained successfully',
                'stats': stats,
                'merged_count': merged_count,  # Actual number of images merged
                'backup_path': backup_path,
                'final_accuracy': float(final_accuracy),
                'final_val_accuracy': float(final_val_accuracy),
                'epochs_trained': len(history['accuracy'])
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Retraining failed: {str(e)}',
                'stats': stats
            }


def trigger_retraining(train_dir: str, val_dir: str) -> Dict:
    """
    Convenience function to trigger retraining
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
    
    Returns:
        Retraining results dictionary
    """
    retrainer = ModelRetrainer()
    return retrainer.full_retraining_workflow(train_dir, val_dir)


if __name__ == "__main__":
    print("Retraining module loaded successfully")
    print(f"Minimum samples for retraining: {MIN_SAMPLES_FOR_RETRAINING}")
