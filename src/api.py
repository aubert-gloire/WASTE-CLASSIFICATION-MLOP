"""
FastAPI Backend for Waste Classification ML Pipeline
Provides REST API endpoints for prediction, upload, retraining, and monitoring
"""
import os
import sys
import time
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MODEL_PATH, CLASS_NAMES, IMAGE_SIZE, 
    UPLOAD_DIR, MODEL_DIR, DATA_DIR
)
from src.prediction import WasteClassifier
from src.retraining import ModelRetrainer

# Initialize FastAPI app
app = FastAPI(
    title="Waste Classification API",
    description="ML Pipeline API for automated waste classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
classifier = None
start_time = time.time()
prediction_count = 0
retraining_status = {"is_retraining": False, "progress": 0, "message": ""}

# Pydantic models
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    uptime_formatted: str
    model_loaded: bool
    prediction_count: int
    cpu_percent: float
    memory_percent: float
    timestamp: str

class RetrainingRequest(BaseModel):
    train_dir: str
    val_dir: str
    epochs: Optional[int] = 5

class RetrainingStatusResponse(BaseModel):
    is_retraining: bool
    progress: int
    message: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global classifier
    try:
        if MODEL_PATH.exists():
            classifier = WasteClassifier(str(MODEL_PATH))
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}")
            print("  Please train the model first using the Jupyter notebook")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Waste Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "upload": "/upload",
            "retrain": "/retrain",
            "retrain_status": "/retrain/status",
            "metrics": "/metrics",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system metrics"""
    uptime_seconds = time.time() - start_time
    uptime_formatted = format_uptime(uptime_seconds)
    
    return HealthResponse(
        status="healthy" if classifier is not None else "model_not_loaded",
        uptime_seconds=uptime_seconds,
        uptime_formatted=uptime_formatted,
        model_loaded=classifier is not None,
        prediction_count=prediction_count,
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=psutil.virtual_memory().percent,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict waste class for uploaded image
    
    Args:
        file: Image file to classify
    
    Returns:
        Prediction results
    """
    global prediction_count
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        result = classifier.predict(image)
        
        # Increment counter
        prediction_count += 1
        
        return PredictionResponse(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            all_probabilities=result['all_probabilities'],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict waste classes for multiple images
    
    Args:
        files: List of image files
    
    Returns:
        List of prediction results
    """
    global prediction_count
    
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            result = classifier.predict(image)
            results.append({
                "filename": file.filename,
                "predicted_class": result['predicted_class'],
                "confidence": result['confidence']
            })
            
            prediction_count += 1
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"predictions": results, "total": len(files)}


@app.post("/upload")
async def upload_data(
    files: List[UploadFile] = File(...),
    class_name: Optional[str] = None
):
    """
    Upload new images for retraining
    
    Args:
        files: List of image files
        class_name: Target class for images
    
    Returns:
        Upload status
    """
    if class_name and class_name not in CLASS_NAMES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid class name. Must be one of: {CLASS_NAMES}"
        )
    
    uploaded_count = 0
    errors = []
    
    for file in files:
        try:
            # Determine class from filename or parameter
            if class_name:
                target_class = class_name
            else:
                # Try to extract from filename
                target_class = None
                for cn in CLASS_NAMES:
                    if cn in file.filename.lower():
                        target_class = cn
                        break
                
                if not target_class:
                    raise ValueError("Cannot determine class from filename")
            
            # Create directory
            upload_class_dir = UPLOAD_DIR / target_class
            upload_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path = upload_class_dir / file.filename
            contents = await file.read()
            
            with open(file_path, 'wb') as f:
                f.write(contents)
            
            uploaded_count += 1
        
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "uploaded": uploaded_count,
        "total": len(files),
        "errors": errors if errors else None
    }


@app.post("/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    request: Optional[RetrainingRequest] = None
):
    """
    Trigger model retraining
    
    Args:
        request: Retraining configuration
    
    Returns:
        Retraining status
    """
    global retraining_status
    
    if retraining_status["is_retraining"]:
        return JSONResponse(
            status_code=409,
            content={"message": "Retraining already in progress"}
        )
    
    # Set default paths if not provided
    if request is None:
        train_dir = str(DATA_DIR / "organized_data" / "train")
        val_dir = str(DATA_DIR / "organized_data" / "validation")
    else:
        train_dir = request.train_dir
        val_dir = request.val_dir
    
    # Check if directories exist, create if needed (for production with uploaded data only)
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    if not train_path.exists() or not val_path.exists():
        print("Training directories don't exist. Creating from uploaded data...")
        # Create directories
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
        
        # Create class subdirectories
        for class_name in CLASS_NAMES:
            (train_path / class_name).mkdir(exist_ok=True)
            (val_path / class_name).mkdir(exist_ok=True)
        
        print("✓ Training directories created. Will use uploaded data for retraining.")
    
    # Start retraining in background
    background_tasks.add_task(
        run_retraining,
        train_dir,
        val_dir
    )
    
    retraining_status = {
        "is_retraining": True,
        "progress": 0,
        "message": "Retraining started"
    }
    
    return {
        "message": "Retraining started in background",
        "status": retraining_status
    }


@app.get("/retrain/status", response_model=RetrainingStatusResponse)
async def get_retraining_status():
    """Get current retraining status"""
    return RetrainingStatusResponse(**retraining_status)


async def run_retraining(train_dir: str, val_dir: str):
    """Background task for model retraining"""
    global retraining_status, classifier
    
    try:
        retraining_status = {
            "is_retraining": True,
            "progress": 10,
            "message": "Checking requirements..."
        }
        
        retrainer = ModelRetrainer()
        
        # Check requirements
        can_retrain, stats = retrainer.check_retraining_requirements()
        
        if not can_retrain:
            retraining_status = {
                "is_retraining": False,
                "progress": 0,
                "message": f"Insufficient data: {stats['total_uploaded']} images"
            }
            return
        
        retraining_status["progress"] = 30
        retraining_status["message"] = "Backing up current model..."
        
        # Backup and merge
        retrainer.backup_current_model()
        retrainer.merge_uploaded_data(train_dir)
        
        retraining_status["progress"] = 50
        retraining_status["message"] = "Training model..."
        
        # Retrain
        result = retrainer.full_retraining_workflow(train_dir, val_dir)
        
        if result['success']:
            # Reload classifier
            classifier = WasteClassifier(str(MODEL_PATH))
            
            retraining_status = {
                "is_retraining": False,
                "progress": 100,
                "message": f"Retraining complete! Accuracy: {result['final_val_accuracy']:.2%}"
            }
        else:
            retraining_status = {
                "is_retraining": False,
                "progress": 0,
                "message": f"Retraining failed: {result['message']}"
            }
    
    except Exception as e:
        retraining_status = {
            "is_retraining": False,
            "progress": 0,
            "message": f"Error: {str(e)}"
        }


@app.get("/metrics")
async def get_metrics():
    """Get system and model metrics"""
    uptime_seconds = time.time() - start_time
    
    # Count uploaded images
    uploaded_count = 0
    if UPLOAD_DIR.exists():
        for class_dir in UPLOAD_DIR.iterdir():
            if class_dir.is_dir():
                uploaded_count += len(list(class_dir.glob('*')))
    
    return {
        "system": {
            "uptime_seconds": uptime_seconds,
            "uptime_formatted": format_uptime(uptime_seconds),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3)
        },
        "model": {
            "loaded": classifier is not None,
            "prediction_count": prediction_count,
            "uploaded_images": uploaded_count
        },
        "retraining": retraining_status,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return classifier.get_model_info()


def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
