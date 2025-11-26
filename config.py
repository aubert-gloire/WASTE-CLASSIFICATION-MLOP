"""
Configuration management for the Waste Classification ML Pipeline
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "waste_classifier_mobilenetv2")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "6"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Training Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
VALIDATION_SPLIT = 0.2

# Data Paths
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploaded"))
LOG_DIR = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "logs"))

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, UPLOAD_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Paths - Allow override from environment
if os.getenv("MODEL_PATH"):
    # Use absolute path from environment variable
    MODEL_PATH = Path(os.getenv("MODEL_PATH"))
    if not MODEL_PATH.is_absolute():
        # If relative path, resolve from PROJECT_ROOT
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH")
else:
    MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.h5"

MODEL_HISTORY_PATH = MODEL_DIR / f"{MODEL_NAME}_{MODEL_VERSION}_history.pkl"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "waste-classification-bucket")
ECR_REPOSITORY = os.getenv("ECR_REPOSITORY", "waste-classification-repo")

# Monitoring
ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))

# Data Augmentation Settings (Enhanced for imbalanced dataset)
AUGMENTATION_CONFIG = {
    "rotation_range": 30,          # Increased from 20
    "width_shift_range": 0.25,     # Increased from 0.2
    "height_shift_range": 0.25,    # Increased from 0.2
    "shear_range": 0.25,           # Increased from 0.2
    "zoom_range": 0.3,             # Increased from 0.2
    "horizontal_flip": True,
    "vertical_flip": True,         # Added vertical flip
    "brightness_range": [0.8, 1.2], # Added brightness variation
    "fill_mode": "nearest"
}

# Retraining Thresholds
MIN_SAMPLES_FOR_RETRAINING = 20
RETRAIN_ACCURACY_THRESHOLD = 0.85

# Retraining Mode - Optimized for quick improvement
RETRAINING_MODE = "fast"  # "fast" = quick fine-tuning
RETRAIN_EPOCHS = 5  # Balanced epochs for model improvement
RETRAIN_BATCH_SIZE = 32  # Standard batch size
DEMO_SAMPLES_PER_CLASS = 100  # Images per class in demo mode (600 total for 6 classes)

print("Configuration loaded")
print(f"  - Model: {MODEL_NAME}")
print(f"  - Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  - Classes: {NUM_CLASSES}")
print(f"  - Epochs: {EPOCHS}")
