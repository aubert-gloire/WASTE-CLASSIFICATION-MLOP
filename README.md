# Waste Classification ML Pipeline

**African Leadership University - Machine Learning Assignment**

Automated waste classification system using MobileNetV2 to classify waste into 6 categories: Cardboard, Glass, Metal, Paper, Plastic, and Trash.

---

## Live Demo

- **Video:** [YouTube Demo](YOUR_YOUTUBE_LINK)
- **API:** https://waste-classification-api-atei.onrender.com
- **UI:** https://waste-classification-ui.onrender.com
- **GitHub:** [Repository](https://github.com/aubert-gloire/WASTE-CLASSIFICATION-MLOP)

---

## Problem & Solution

**Challenge:** Manual waste sorting is slow, expensive, and error-prone. 25-30% of recycling is contaminated and rejected.

**Solution:** AI-powered waste classification system that:
- Classifies waste instantly with 89.2% accuracy
- Processes images in <1 second
- Scales to handle high demand
- Continuously improves through retraining

**Impact:** Increase recycling rate from 30% to 60%+, reduce contamination by 70-80%

---

## Features

- Single image prediction with confidence scores
- 3+ interactive data visualizations
- Bulk image upload for retraining
- One-click model retraining (demo: 2 min, full: 30-60 min)
- Real-time system monitoring
- Load testing with Locust

---

## Project Structure

```
├── notebook/
│   └── waste_classification.ipynb    # ML pipeline
├── src/
│   ├── api.py                         # FastAPI backend
│   ├── model.py                       # Model training
│   ├── prediction.py                  # Inference
│   └── retraining.py                  # Retraining logic
├── ui/
│   └── app.py                         # Streamlit UI
├── models/
│   └── waste_classifier_mobilenetv2_v1.h5
├── data/                              # Training data
├── docker/                            # Containerization
├── locustfile.py                      # Load testing
├── requirements.txt
└── render.yaml                        # Deployment config
```

---

## Quick Start

### Local Setup

```bash
# Clone and install
git clone https://github.com/aubert-gloire/WASTE-CLASSIFICATION-MLOP.git
cd WASTE-CLASSIFICATION-MLOP
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train model (run notebook)
jupyter notebook notebook/waste_classification.ipynb

# Start API
python src/api.py

# Start UI (new terminal)
streamlit run ui/app.py
```

Access at `http://localhost:8501`

### Cloud Deployment

```bash
# Push to GitHub
git init
git add .
git commit -m "Initial commit"
git push origin main

# Deploy on Render
# 1. Sign up at render.com
# 2. Click "New" → "Blueprint"
# 3. Connect GitHub repo
# 4. Wait 5-10 minutes
```

---

## Usage

### Predict via UI
1. Go to "Predict" page
2. Upload waste image
3. View classification and confidence

### Predict via API
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

### Retrain Model
1. Upload 20+ images via "Upload Data" page
2. Go to "Retrain Model" page
3. Click "Start Retraining"
4. Monitor progress

---

## Model Information

**Architecture:** MobileNetV2 (transfer learning)
**Input:** 224x224 RGB images
**Output:** 6 classes
**Training:** 10 epochs, batch size 32

### Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 89.2%  |
| Precision | 88.5%  |
| Recall    | 87.8%  |
| F1-Score  | 88.1%  |

### Per-Class Performance

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Cardboard | 0.85      | 0.83   | 0.84     |
| Glass     | 0.92      | 0.89   | 0.90     |
| Metal     | 0.88      | 0.86   | 0.87     |
| Paper     | 0.90      | 0.91   | 0.90     |
| Plastic   | 0.87      | 0.88   | 0.87     |
| Trash     | 0.83      | 0.80   | 0.81     |

---

## Load Testing

**Tool:** Locust
**Test:** 50 concurrent users, 5 min duration

### Results (Single Instance)

| Metric | Value |
|--------|-------|
| Total Requests | 697 |
| RPS | 2.34 |
| Avg Response Time | 18.2s |
| Median Response Time | 9.8s |
| Failure Rate | 13.6% |

**Note:** Deploy with multiple instances on Render to improve performance.

### Run Tests

```bash
# Update locustfile.py with deployed URL
locust -f locustfile.py --host https://your-api.onrender.com
```

---

## Technologies

**ML:** TensorFlow/Keras, MobileNetV2, scikit-learn, OpenCV  
**Backend:** FastAPI, Uvicorn  
**Frontend:** Streamlit, Plotly  
**DevOps:** Docker, Render  
**Testing:** Locust

---

## Contributors

Your Name  
African Leadership University  
Machine Learning Engineering

---

## License

Academic assignment for African Leadership University

---

## Acknowledgments

- Dataset: Garbage Classification (Kaggle)
- African Leadership University
- TensorFlow Team

---

Built for a cleaner planet
