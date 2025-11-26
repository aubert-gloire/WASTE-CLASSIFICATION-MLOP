"""
Streamlit UI for Waste Classification ML Pipeline
Provides user interface for prediction, visualization, upload, and retraining
"""
import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from pathlib import Path
import sys

# Page configuration
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Add parent directory to path
# __file__ is ui/app.py, parent is ui/, parent.parent is project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import CLASS_NAMES, DATA_DIR, MODEL_DIR
except Exception as e:
    CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    DATA_DIR = project_root / "data"
    MODEL_DIR = project_root / "models"

# Define dataset stats functions directly in UI
import pickle
import json

# Compatibility helper for st.image across Streamlit versions
def display_image(img, caption=None):
    """Display an image using the available Streamlit parameter.

    Tries `use_container_width`, falls back to `use_column_width`, then no width arg.
    """
    try:
        if caption is not None:
            st.image(img, caption=caption, use_container_width=True)
        else:
            st.image(img, use_container_width=True)
        return
    except TypeError:
        pass

    try:
        if caption is not None:
            st.image(img, caption=caption, use_column_width=True)
        else:
            st.image(img, use_column_width=True)
        return
    except TypeError:
        pass

    # Final fallback
    if caption is not None:
        st.image(img, caption=caption)
    else:
        st.image(img)


def display_plotly(fig):
    """Display a Plotly figure with compatibility across Streamlit versions.

    Tries `use_container_width=True`, falls back to a simple `st.plotly_chart(fig)`.
    """
    try:
        st.plotly_chart(fig, use_container_width=True)
        return
    except TypeError:
        pass

    try:
        # Some Streamlit versions accept `use_container_width` but as a keyword
        st.plotly_chart(fig)
        return
    except Exception:
        # Final fallback: attempt without extra args
        st.plotly_chart(fig)
def get_dataset_statistics():
    """Get real dataset statistics"""
    stats = {'class_counts': {}, 'total_train': 0, 'total_val': 0, 'total_images': 0}
    
    organized_data_path = DATA_DIR / "organized_data"
    if not organized_data_path.exists():
        return {**stats, 'available': False}
    
    train_dir = organized_data_path / "train"
    val_dir = organized_data_path / "validation"
    
    for class_name in CLASS_NAMES:
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        train_count = len(list(train_class_dir.glob('*.jpg')) + list(train_class_dir.glob('*.png')) + list(train_class_dir.glob('*.jpeg'))) if train_class_dir.exists() else 0
        val_count = len(list(val_class_dir.glob('*.jpg')) + list(val_class_dir.glob('*.png')) + list(val_class_dir.glob('*.jpeg'))) if val_class_dir.exists() else 0
        
        stats['class_counts'][class_name] = train_count + val_count
        stats['total_train'] += train_count
        stats['total_val'] += val_count
    
    stats['total_images'] = stats['total_train'] + stats['total_val']
    stats['available'] = True
    return stats

def get_model_metrics():
    """Load training history"""
    history_path = MODEL_DIR / "training_history.pkl"
    if not history_path.exists():
        return {'available': False}
    
    try:
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        metrics = {
            'accuracy': history['accuracy'][-1],
            'val_accuracy': history['val_accuracy'][-1],
            'loss': history['loss'][-1],
            'val_loss': history['val_loss'][-1],
            'precision': history.get('precision', [0])[-1],
            'val_precision': history.get('val_precision', [0])[-1],
            'recall': history.get('recall', [0])[-1],
            'val_recall': history.get('val_recall', [0])[-1],
            'epochs_trained': len(history['accuracy']),
            'available': True
        }
        
        if metrics['val_precision'] > 0 and metrics['val_recall'] > 0:
            metrics['val_f1_score'] = 2 * (metrics['val_precision'] * metrics['val_recall']) / (metrics['val_precision'] + metrics['val_recall'])
        else:
            metrics['val_f1_score'] = 0
        
        return metrics
    except:
        return {'available': False}

def get_per_class_performance():
    """Load per-class performance"""
    report_path = MODEL_DIR / "classification_report.json"
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def get_uploaded_data_stats():
    """Get uploaded data stats"""
    upload_dir = DATA_DIR / "uploaded"
    stats = {'total_uploaded': 0, 'by_class': {}, 'ready_for_retraining': False}
    
    if not upload_dir.exists():
        return stats
    
    for class_name in CLASS_NAMES:
        class_dir = upload_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg')))
            stats['by_class'][class_name] = count
            stats['total_uploaded'] += count
    
    stats['ready_for_retraining'] = stats['total_uploaded'] >= 20
    return stats


# Custom CSS - Dark Grey Theme
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: #e8e8e8;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #a8b2d1;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #2a2d3a 0%, #1f222e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #3a3f52;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Prediction Box - Enhanced */
    .prediction-box {
        border: 2px solid #00d4ff;
        padding: 2.5rem;
        border-radius: 20px;
        background: linear-gradient(145deg, #0f3460 0%, #1a1a2e 100%);
        margin: 1.5rem 0;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3), 0 10px 25px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(0, 212, 255, 0.1) 50%, transparent 70%);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .prediction-class {
        color: #00ff88;
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
    }
    
    .prediction-confidence {
        color: #00d4ff;
        font-size: 2rem;
        margin: 0.5rem 0;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        position: relative;
        z-index: 1;
    }
    
    /* Category Cards */
    .category-card {
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 1.5rem 1rem;
        background: linear-gradient(145deg, #2a2d3a 0%, #1f222e 100%);
        border-radius: 15px;
        border: 2px solid #3a3f52;
        color: #e8e8e8;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    .category-card:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4), 0 8px 16px rgba(0, 0, 0, 0.4);
    }
    
    /* Info/Success/Warning/Error Boxes */
    .stAlert {
        background: rgba(42, 45, 58, 0.8);
        border-left: 4px solid #00d4ff;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #0096c7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.4);
        transform: translateY(-2px);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(42, 45, 58, 0.5);
        border: 2px dashed #3a3f52;
        border-radius: 15px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #00d4ff;
        background: rgba(0, 212, 255, 0.05);
    }
    
    /* Selectbox/Input */
    .stSelectbox, .stTextInput {
        color: #e8e8e8;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(42, 45, 58, 0.3);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(31, 34, 46, 0.5);
        border-radius: 8px;
        color: #a8b2d1;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0096c7 100%);
        color: white;
        box-shadow: 0 4px 10px rgba(0, 212, 255, 0.3);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%);
        box-shadow: 0 2px 10px rgba(0, 212, 255, 0.5);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(42, 45, 58, 0.6);
        border-radius: 10px;
        color: #e8e8e8;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(0, 212, 255, 0.1);
        border-color: #00d4ff;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00ff88;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a8b2d1;
        font-weight: 600;
    }
    
    /* General Text */
    p, li, span {
        color: #ccd6f6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e8e8e8;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff 0%, #0096c7 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00ff88 0%, #00d4ff 100%);
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_metrics():
    """Fetch metrics from API"""
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def predict_image(image_file):
    """Send image to API for prediction"""
    try:
        # Ensure we send a proper file-tuple to requests
        try:
            # If Streamlit's UploadedFile-like object, read bytes
            image_file.seek(0)
            content = image_file.read()
            filename = getattr(image_file, 'name', 'upload.jpg')
            content_type = getattr(image_file, 'type', 'image/jpeg')
        except Exception:
            # If a raw bytes/BytesIO was passed
            if isinstance(image_file, (bytes, bytearray)):
                content = bytes(image_file)
                filename = 'upload.jpg'
                content_type = 'image/jpeg'
            else:
                content = None
                filename = 'upload.jpg'
                content_type = 'image/jpeg'

        if content is None:
            return {"error": "Could not read uploaded image"}

        files = {"file": (filename, io.BytesIO(content), content_type)}
        response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Prediction failed")}
    except Exception as e:
        return {"error": str(e)}


def upload_images(files, class_name):
    """Upload multiple images for retraining"""
    try:
        files_data = [("files", (f.name, f, "image/jpeg")) for f in files]
        params = {"class_name": class_name} if class_name else {}
        
        response = requests.post(
            f"{API_URL}/upload",
            files=files_data,
            params=params
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Upload failed")}
    except Exception as e:
        return {"error": str(e)}


def trigger_retraining():
    """Trigger model retraining"""
    try:
        response = requests.post(f"{API_URL}/retrain")
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_retraining_status():
    """Get retraining status"""
    try:
        response = requests.get(f"{API_URL}/retrain/status")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


# Main App
def main():
    # Header
    st.markdown('<div class="main-header">Waste Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Waste Classification using Machine Learning</div>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("API is not running. Please start the API server first.")
        st.code("python src/api.py", language="bash")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Predict", "Visualizations", "Upload Data", "Retrain Model", "Monitoring"]
    )
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Predict":
        show_prediction_page()
    elif page == "Visualizations":
        show_visualization_page()
    elif page == "Upload Data":
        show_upload_page()
    elif page == "Retrain Model":
        show_retrain_page()
    elif page == "Monitoring":
        show_monitoring_page()


def show_home_page():
    """Home page with project overview"""
    st.header("Welcome to the Waste Classification System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Problem**\n\nManual waste sorting is slow, expensive, and error-prone. 25-30% of recycling is contaminated.")
    
    with col2:
        st.success("**Solution**\n\nAutomated waste classification system with 6 categories and real-time predictions.")
    
    with col3:
        st.warning("**Impact**\n\nIncrease recycling rate from 30% to 60%+. Reduce contamination by 70-80%.")
    
    st.markdown("---")
    
    # System status
    st.subheader("System Status")
    metrics = get_metrics()
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Uptime", metrics['system']['uptime_formatted'])
        
        with col2:
            st.metric("CPU Usage", f"{metrics['system']['cpu_percent']:.1f}%")
        
        with col3:
            st.metric("Memory Usage", f"{metrics['system']['memory_percent']:.1f}%")
        
        with col4:
            st.metric("Predictions Made", metrics['model']['prediction_count'])
    
    st.markdown("---")
    
    # Features
    st.subheader("Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Single Image Prediction
        - Upload any waste image
        - Get instant classification
        - View confidence scores
        - See all class probabilities
        
        ### Data Visualizations
        - Class distribution analysis
        - Prediction statistics
        - Model performance metrics
        - Interactive charts
        """)
    
    with col2:
        st.markdown("""
        ### Bulk Data Upload
        - Upload multiple images
        - Organize by waste type
        - Prepare for retraining
        - Track upload progress
        
        ### Model Retraining
        - One-click retraining
        - Progress monitoring
        - Performance comparison
        - Automatic model update
        """)
    
    st.markdown("---")
    
    # Waste categories
    st.subheader("Supported Waste Categories")
    
    cols = st.columns(6)
    categories = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
    
    for col, cat in zip(cols, categories):
        with col:
            st.markdown(f"<div class='category-card'>{cat}</div>", unsafe_allow_html=True)


def show_prediction_page():
    """Prediction page for single image classification"""
    st.header("Waste Classification Prediction")
    
    st.markdown("""
    Upload an image of waste to classify it into one of 6 categories:
    **Cardboard, Glass, Metal, Paper, Plastic, or Trash**
    """)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of waste item"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])

        # Read bytes once and reuse
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        file_bytes = uploaded_file.read()

        with col1:
            st.subheader("Uploaded Image")
            try:
                image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
                display_image(image)
            except Exception as e:
                st.error(f"Failed to open image for display: {e}")
                # Fallback: streamlit can accept raw bytes too
                try:
                    display_image(file_bytes)
                except Exception as e2:
                    st.error(f"Fallback display also failed: {e2}")

        with col2:
            st.subheader("Prediction Result")

            with st.spinner("Classifying..."):
                # Send the read bytes to the prediction endpoint
                result = predict_image(io.BytesIO(file_bytes) if not isinstance(file_bytes, (bytes, bytearray)) else file_bytes)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                confidence = result['confidence']
                predicted_class = result['predicted_class']
                
                # Confidence threshold check
                CONFIDENCE_THRESHOLD = 0.60
                
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f"Low Confidence Prediction ({confidence*100:.2f}%)")
                    st.info("""
                    **Manual Review Recommended**
                    
                    The model's confidence is below 60%, which suggests:
                    - The item may be ambiguous or damaged
                    - The model needs more training data for this type
                    - Consider verifying the classification manually
                    """)
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 class="prediction-class">
                        {predicted_class.upper()}
                    </h2>
                    <p class="prediction-confidence">
                        Confidence: {confidence*100:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar with color coding
                if confidence >= 0.80:
                    st.success("High confidence prediction")
                elif confidence >= CONFIDENCE_THRESHOLD:
                    st.info("Moderate confidence prediction")
                else:
                    st.warning("Low confidence - manual review recommended")
                    
                st.progress(confidence)
                
                # All probabilities
                st.subheader("All Class Probabilities")
                probs_df = pd.DataFrame({
                    'Class': list(result['all_probabilities'].keys()),
                    'Probability': list(result['all_probabilities'].values())
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(
                    probs_df,
                    x='Probability',
                    y='Class',
                    orientation='h',
                    color='Probability',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=400)
                display_plotly(fig)
                
                # Recycling tip
                st.info(f"**Recycling Tip for {result['predicted_class'].capitalize()}:**\n\n" + 
                       get_recycling_tip(result['predicted_class']))


def show_visualization_page():
    """Visualization page with dataset insights"""
    st.header("Data Visualizations & Insights")
    
    # Load real dataset statistics
    dataset_stats = get_dataset_statistics()
    model_metrics = get_model_metrics()
    per_class_perf = get_per_class_performance()
    uploaded_stats = get_uploaded_data_stats()
    
    # Show data availability status
    if not dataset_stats.get('available', False):
        st.warning("Dataset not yet organized. Please run the Jupyter notebook first to train the model.")
        st.info("After training, real dataset statistics will be displayed here.")
        return
    
    st.success(f"Showing data from trained model (Total: {dataset_stats['total_images']} images)")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Class Distribution", "Model Performance", "Prediction Stats", "Upload Status"])
    
    with tab1:
        st.subheader("Class Distribution Analysis")
        st.markdown("""
        Shows the number of images of each waste type in the training dataset.
        A balanced distribution helps the model learn all classes equally well.
        """)
        
        # Data from dataset
        class_counts = dataset_stats['class_counts']
        
        df = pd.DataFrame({
            'Class': list(class_counts.keys()),
            'Count': list(class_counts.values())
        })
        
                fig = px.bar(
                    df,
                    x='Class',
                    y='Count',
                    title='Waste Type Distribution in Training Dataset',
                    color='Count',
                    color_continuous_scale='Viridis',
                    text='Count'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(height=500)
            display_plotly(fig)
        
        # Additional statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Images", f"{sum(class_counts.values())}")
        with col2:
            st.metric("Training Set", f"{dataset_stats['total_train']}")
        with col3:
            st.metric("Validation Set", f"{dataset_stats['total_val']}")
        
        st.markdown(f"""
        **Dataset Insights:**
        - **Total images:** {sum(class_counts.values())}
        - **Most common:** {max(class_counts, key=class_counts.get)} ({max(class_counts.values())} images)
        - **Least common:** {min(class_counts, key=class_counts.get)} ({min(class_counts.values())} images)
        - **Imbalance ratio:** {max(class_counts.values()) / max(1, min(class_counts.values())):.2f}:1
        - **Average per class:** {sum(class_counts.values()) / len(class_counts):.0f} images
        """)
    
    with tab2:
        st.subheader("Model Performance Metrics")
        st.markdown("Performance metrics from the trained MobileNetV2 model on validation data.")
        
        if model_metrics and model_metrics.get('available'):
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{model_metrics['val_accuracy']*100:.2f}%")
            with col2:
                st.metric("Precision", f"{model_metrics['val_precision']*100:.2f}%")
            with col3:
                st.metric("Recall", f"{model_metrics['val_recall']*100:.2f}%")
            with col4:
                st.metric("F1-Score", f"{model_metrics['val_f1_score']*100:.2f}%")
            
            st.success(f"Model trained for {model_metrics['epochs_trained']} epochs")
            
            # Training history
            st.markdown("#### Training Progress")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Training Loss", f"{model_metrics['loss']:.4f}")
                st.metric("Final Validation Loss", f"{model_metrics['val_loss']:.4f}")
            with col2:
                st.metric("Training Accuracy", f"{model_metrics['accuracy']*100:.2f}%")
                st.metric("Validation Accuracy", f"{model_metrics['val_accuracy']*100:.2f}%")
        else:
            st.warning("Model metrics not available. Train the model first.")
            return
        
        # Per-class performance from actual model
        if per_class_perf:
            st.markdown("#### Per-Class Performance")
            per_class_data = pd.DataFrame([
                {
                    'Class': class_name,
                    'Precision': per_class_perf[class_name]['precision'],
                    'Recall': per_class_perf[class_name]['recall'],
                    'F1-Score': per_class_perf[class_name]['f1-score']
                }
                for class_name in CLASS_NAMES
            ])
        else:
            st.info("Using estimated per-class metrics based on overall performance")
            per_class_data = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Precision': [model_metrics['val_precision']] * len(CLASS_NAMES),
                'Recall': [model_metrics['val_recall']] * len(CLASS_NAMES),
                'F1-Score': [model_metrics['val_f1_score']] * len(CLASS_NAMES)
            })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Precision', x=per_class_data['Class'], y=per_class_data['Precision'],
                            text=[f"{v:.2%}" for v in per_class_data['Precision']], textposition='outside'))
        fig.add_trace(go.Bar(name='Recall', x=per_class_data['Class'], y=per_class_data['Recall'],
                            text=[f"{v:.2%}" for v in per_class_data['Recall']], textposition='outside'))
        fig.add_trace(go.Bar(name='F1-Score', x=per_class_data['Class'], y=per_class_data['F1-Score'],
                            text=[f"{v:.2%}" for v in per_class_data['F1-Score']], textposition='outside'))
        
        fig.update_layout(
            title='Per-Class Performance Metrics',
            barmode='group',
            height=500,
            yaxis_title='Score',
            xaxis_title='Waste Category'
        )
        display_plotly(fig)
        
        st.info("These metrics are calculated from model evaluation on the validation set.")
    
    with tab3:
        st.subheader("API Prediction Statistics")
        
        metrics = get_metrics()
        
        if metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions Made", metrics['model']['prediction_count'])
                st.metric("Uploaded Images (Pending)", metrics['model']['uploaded_images'])
            
            with col2:
                st.metric("Model Status", "Loaded" if metrics['model']['loaded'] else "Not Loaded")
                st.metric("System Uptime", metrics['system']['uptime_formatted'])
            
            with col3:
                st.metric("CPU Usage", f"{metrics['system']['cpu_percent']:.1f}%")
                st.metric("Memory Usage", f"{metrics['system']['memory_percent']:.1f}%")
            
            st.success("Statistics from running API server")
        else:
            st.warning("Could not fetch metrics from API. Make sure the API server is running.")
    
    with tab4:
        st.subheader("Upload Status & Preprocessing Info")
        st.markdown("""
        Monitor the status of uploaded images ready for retraining.
        This shows the data preprocessing pipeline for new images.
        """)
        
        if uploaded_stats['total_uploaded'] > 0:
            st.success(f"{uploaded_stats['total_uploaded']} images uploaded and ready for retraining")
            
            # Show distribution of uploaded data
            uploaded_df = pd.DataFrame([
                {'Class': class_name, 'Uploaded': count}
                for class_name, count in uploaded_stats['by_class'].items()
                if count > 0
            ])
            
            if not uploaded_df.empty:
                fig = px.bar(
                    uploaded_df,
                    x='Class',
                    y='Uploaded',
                    title='Uploaded Images by Class (Pending Retraining)',
                    color='Uploaded',
                    text='Uploaded'
                )
                fig.update_traces(textposition='outside')
                display_plotly(fig)
            
            # Retraining readiness
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Uploaded", uploaded_stats['total_uploaded'])
            with col2:
                ready = "Yes" if uploaded_stats['ready_for_retraining'] else "No"
                st.metric("Ready for Retraining", ready)
            
            if uploaded_stats['ready_for_retraining']:
                st.success("Sufficient data for retraining. Go to 'Retrain Model' page.")
            else:
                needed = 20 - uploaded_stats['total_uploaded']
                st.info(f"Upload {needed} more images to enable retraining (minimum 20 required)")
            
            # Show preprocessing steps that will be applied
            with st.expander("Preprocessing Steps for Uploaded Data"):
                st.markdown("""
                When you trigger retraining, uploaded images will undergo:
                
                1. **Validation:** Check file formats (jpg, png, jpeg)
                2. **Resizing:** Resize to 224x224 pixels
                3. **Normalization:** Scale pixel values to [0, 1]
                4. **Augmentation:** Apply random transformations:
                   - Rotation (±20°)
                   - Width/Height shift (±20%)
                   - Horizontal flip
                   - Zoom (±20%)
                5. **Integration:** Merge with existing training data
                6. **Backup:** Save current model before retraining
                7. **Retraining:** Fine-tune model with combined dataset
                """)
        else:
            st.info("No uploaded images yet. Go to 'Upload Data' page to add new training data.")
            st.markdown("""
    **How to prepare data for retraining:**
    1. Navigate to the 'Upload Data' page
    2. Select the waste category
    3. Upload clear, well-lit images
    4. Minimum 20 images required across all classes
    5. Images will be automatically preprocessed
    """)
def show_upload_page():
    """Upload page for bulk data upload"""
    st.header("Upload New Training Data")
    
    st.markdown("""
    Upload multiple images to expand the training dataset. These images will be used when you trigger retraining.
    
    **Requirements:**
    - Upload at least 20 images total for retraining
    - Images should be clear and well-lit
    - Select the correct waste category
    """)
    
    # Select class
    selected_class = st.selectbox(
        "Select Waste Category",
        CLASS_NAMES,
        help="Choose which category these images belong to"
    )
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="You can upload multiple images at once"
    )
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} images for class: **{selected_class}**")
        
        # Show preview
        with st.expander("Preview Images"):
            cols = st.columns(5)
            for idx, file in enumerate(uploaded_files[:10]):  # Show first 10
                with cols[idx % 5]:
                    image = Image.open(file)
                    display_image(image, caption=file.name)
            
            if len(uploaded_files) > 10:
                st.info(f"...and {len(uploaded_files) - 10} more images")
        
        # Upload button
        if st.button("Upload Images", type="primary"):
            with st.spinner("Uploading..."):
                # Reset file pointers
                for f in uploaded_files:
                    f.seek(0)
                
                result = upload_images(uploaded_files, selected_class)
            
            if "error" in result:
                st.error(f"Upload failed: {result['error']}")
            else:
                st.success(f"Successfully uploaded {result['uploaded']} images")
                
                if result.get('errors'):
                    with st.expander("View Errors"):
                        for error in result['errors']:
                            st.warning(f"{error['filename']}: {error['error']}")
    
    # Show current upload status
    st.markdown("---")
    st.subheader("Current Upload Status")
    
    metrics = get_metrics()
    if metrics:
        uploaded_count = metrics['model']['uploaded_images']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Uploaded Images", uploaded_count)
        with col2:
            required = 20
            ready = uploaded_count >= required
            st.metric("Ready for Retraining", "Yes" if ready else f"Need {required - uploaded_count} more")
        
        if uploaded_count > 0:
            st.progress(min(uploaded_count / required, 1.0))


def show_retrain_page():
    """Retraining page"""
    st.header("Model Retraining")
    
    st.markdown("""
    Retrain the model with newly uploaded data to improve its performance.
    
    **Process:**
    1. Uploaded images are merged with existing training data
    2. Model is retrained for 5 additional epochs
    3. New model replaces the old one if performance improves
    4. Previous model is backed up automatically
    """)
    
    # Check retraining status
    status = get_retraining_status()
    
    if status and status['is_retraining']:
        st.warning("Retraining in progress...")
        st.progress(status['progress'] / 100)
        st.info(f"Status: {status['message']}")
        
        # Auto-refresh
        if st.button("Refresh Status"):
            st.rerun()
        
        # Auto-refresh every 5 seconds
        time.sleep(5)
        st.rerun()
    
    else:
        # Show current status
        metrics = get_metrics()
        
        if metrics:
            uploaded_count = metrics['model']['uploaded_images']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Uploaded Images", uploaded_count)
            
            with col2:
                st.metric("Current Predictions", metrics['model']['prediction_count'])
            
            with col3:
                ready = uploaded_count >= 50
                st.metric("Status", "Ready" if ready else "Waiting")
            
            st.markdown("---")
            
            # Retrain button
            if uploaded_count >= 20:
                st.success("Sufficient data available for retraining")
                
                if st.button("Start Retraining", type="primary"):
                    with st.spinner("Initiating retraining..."):
                        result = trigger_retraining()
                    
                    if "error" in result:
                        st.error(f"Failed to start retraining: {result['error']}")
                    else:
                        st.success("Retraining started")
                        time.sleep(2)
                        st.rerun()
            else:
                st.warning(f"Need at least 20 images. Currently have {uploaded_count}.")
                st.info("Go to 'Upload Data' page to add more images")
        
        # Show last retraining result
        if status and status['message'] and not status['is_retraining']:
            st.markdown("---")
            st.subheader("Last Retraining Result")
            
            if "complete" in status['message'].lower():
                st.success(status['message'])
            elif "failed" in status['message'].lower():
                st.error(status['message'])
            else:
                st.info(status['message'])


def show_monitoring_page():
    """Monitoring page with system metrics"""
    st.header("System Monitoring & Uptime")
    
    st.markdown("System performance and model metrics monitoring")
    
    # Get metrics
    metrics = get_metrics()
    
    if not metrics:
        st.error("Could not fetch metrics from API")
        return
    
    # System metrics
    st.subheader("System Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Uptime", metrics['system']['uptime_formatted'])
    
    with col2:
        cpu = metrics['system']['cpu_percent']
        st.metric("CPU Usage", f"{cpu:.1f}%", delta=None)
    
    with col3:
        mem = metrics['system']['memory_percent']
        st.metric("Memory Usage", f"{mem:.1f}%", delta=None)
    
    with col4:
        mem_used = metrics['system']['memory_used_gb']
        st.metric("Memory Used", f"{mem_used:.2f} GB")
    
    # Progress bars
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**CPU Usage**")
        st.progress(cpu / 100)
    
    with col2:
        st.markdown("**Memory Usage**")
        st.progress(mem / 100)
    
    # Model metrics
    st.markdown("---")
    st.subheader("Model Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Status", "Loaded" if metrics['model']['loaded'] else "Not Loaded")
    
    with col2:
        st.metric("Total Predictions", metrics['model']['prediction_count'])
    
    with col3:
        st.metric("Uploaded Images", metrics['model']['uploaded_images'])
    
    # Retraining status
    st.markdown("---")
    st.subheader("Retraining Status")
    
    retrain_status = metrics['retraining']
    
    col1, col2 = st.columns(2)
    
    with col1:
        status_text = "In Progress" if retrain_status['is_retraining'] else "Idle"
        st.metric("Status", status_text)
    
    with col2:
        st.metric("Progress", f"{retrain_status['progress']}%")
    
    if retrain_status['message']:
        st.info(retrain_status['message'])
    
    # Auto-refresh
    st.markdown("---")
    if st.button("Refresh Metrics"):
        st.rerun()
    
    st.info("Metrics update automatically. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def get_recycling_tip(waste_type):
    """Get recycling tip for waste type"""
    tips = {
        'cardboard': "Flatten boxes to save space. Remove any plastic tape or labels before recycling.",
        'glass': "Rinse containers before recycling. Separate by color if your facility requires it.",
        'metal': "Rinse cans and remove labels. Aluminum and steel are highly recyclable.",
        'paper': "Keep paper dry and clean. Staples are usually okay, but remove plastic windows from envelopes.",
        'plastic': "Check the recycling number (1-7). Rinse containers and remove caps.",
        'trash': "This item cannot be recycled. Dispose of it in general waste."
    }
    return tips.get(waste_type, "Check with your local recycling facility for proper disposal.")


if __name__ == "__main__":
    main()
