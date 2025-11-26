"""
Prediction Module
Handles inference and prediction on new images
"""
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGE_SIZE, CLASS_NAMES, MODEL_PATH
from src.preprocessing import load_and_preprocess_image


class WasteClassifier:
    """Waste classification prediction class"""
    
    def __init__(self, model_path: str = str(MODEL_PATH)):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = CLASS_NAMES
        self.image_size = IMAGE_SIZE
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
        
        Returns:
            Preprocessed image array
        """
        if isinstance(image_input, str):
            # Load from file path
            img = load_and_preprocess_image(image_input, (self.image_size, self.image_size))
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to numpy array
            img = np.array(image_input.resize((self.image_size, self.image_size)))
            img = img.astype('float32') / 255.0
        elif isinstance(image_input, np.ndarray):
            # Resize numpy array
            img = cv2.resize(image_input, (self.image_size, self.image_size))
            img = img.astype('float32') / 255.0
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Predict waste class for a single image
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img = self.preprocess_image(image_input)
        
        # Make prediction
        predictions = self.model.predict(img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get all class probabilities
        all_probabilities = {
            self.class_names[i]: float(predictions[0][i])
            for i in range(len(self.class_names))
        }
        
        result = {
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'prediction_array': predictions[0].tolist()
        }
        
        return result
    
    def predict_batch(self, image_inputs: List[Union[str, np.ndarray]]) -> List[Dict]:
        """
        Predict waste classes for multiple images
        
        Args:
            image_inputs: List of image file paths or numpy arrays
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_input in image_inputs:
            try:
                result = self.predict(image_input)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'predicted_class': None,
                    'confidence': 0.0
                })
        
        return results
    
    def predict_with_top_k(self, image_input: Union[str, np.ndarray], k: int = 3) -> Dict:
        """
        Predict with top-k classes
        
        Args:
            image_input: Image file path or numpy array
            k: Number of top predictions to return
        
        Returns:
            Dictionary with top-k predictions
        """
        # Preprocess and predict
        img = self.preprocess_image(image_input)
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        
        top_k_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': float(predictions[idx])
            }
            for idx in top_k_indices
        ]
        
        return {
            'top_predictions': top_k_predictions,
            'predicted_class': top_k_predictions[0]['class'],
            'confidence': top_k_predictions[0]['confidence']
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'class_names': self.class_names,
            'num_classes': len(self.class_names)
        }


def predict_single_image(image_path: str, model_path: str = str(MODEL_PATH)) -> Dict:
    """
    Convenience function to predict a single image
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model
    
    Returns:
        Prediction dictionary
    """
    classifier = WasteClassifier(model_path)
    return classifier.predict(image_path)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Evaluate prediction performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics


if __name__ == "__main__":
    print("Prediction module loaded successfully")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Classes: {CLASS_NAMES}")
