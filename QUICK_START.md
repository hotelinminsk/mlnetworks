# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (if not already trained)

```bash
make train_all
```

### 3. Run Application

```bash
# Option A: Streamlit Dashboard
make app
# → http://localhost:8501

# Option B: FastAPI
make api
# → http://localhost:8000/docs

# Option C: Docker (everything at once)
make docker_compose_up
# → API: http://localhost:8000
# → Dashboard: http://localhost:8501
# → MLflow: http://localhost:5000
```

---

## 📋 Essential Commands

### Training

```bash
make train_all          # Train all 5 models
make train_mlflow       # Train with experiment tracking
make compare            # Generate model comparison report
```

### Applications

```bash
make app                # Streamlit dashboard
make api                # FastAPI (development)
make test_api           # Test API endpoints
make mlflow_ui          # View experiment tracking
```

### Docker

```bash
make docker_compose_up  # Start all services
make docker_logs        # View logs
make docker_compose_down # Stop all services
```

### Help

```bash
make help               # See all commands
```

---

## 🎯 What Each Application Does

### Streamlit Dashboard (Port 8501)

**4 Interactive Pages:**
1. **Model Performance** - ROC curves, metrics, confusion matrices
2. **Data Explorer** - Inspect predictions, filter samples
3. **Live Prediction** - Test with custom traffic samples
4. **Model Explanation** - SHAP feature importance & explanations

### FastAPI (Port 8000)

**Endpoints:**
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /docs` - Interactive API docs

### MLflow UI (Port 5000)

**Features:**
- View all training experiments
- Compare model versions
- Track metrics over time
- Manage model registry

---

## 📊 Model Performance

| Model | ROC AUC | Accuracy | F1 Score |
|-------|---------|----------|----------|
| **Gradient Boosting** 🏆 | **0.9787** | 92.4% | 0.9287 |
| Random Forest | 0.9783 | 92.5% | 0.9292 |
| Extra Trees | 0.9260 | 80.4% | 0.8487 |
| SGD Classifier | 0.9025 | 78.6% | 0.8334 |
| Isolation Forest | 0.7804 | 70.8% | 0.7500 |

---

## 🔍 Example API Usage

### Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "dur": 0.5,
    "spkts": 20,
    "dpkts": 15,
    "sbytes": 2048,
    "dbytes": 1536,
    "rate": 70.0,
    "sload": 32768.0,
    "dload": 24576.0,
    "proto": "tcp",
    "service": "http",
    "state": "FIN"
}

response = requests.post(url, json=data)
print(response.json())
# {'prediction': 'attack', 'probability': 0.95, 'confidence': 'high'}
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"dur":0.5,"spkts":20,"dpkts":15,"sbytes":2048,"dbytes":1536,"rate":70.0,"sload":32768.0,"dload":24576.0,"proto":"tcp","service":"http","state":"FIN"}'
```

---

## 🐳 Docker Quick Reference

```bash
# Build images
make docker_build

# Start services (detached)
docker-compose up -d

# View logs (follow)
docker-compose logs -f

# Check status
docker-compose ps

# Stop services
docker-compose down

# Restart single service
docker-compose restart api

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

---

## 📁 Project Structure

```
mlnetworks/
├── app/                    # Streamlit dashboard
│   └── pages/              # Dashboard pages (4 pages)
├── api/                    # FastAPI application
│   ├── main.py             # API endpoints
│   └── models.py           # Pydantic schemas
├── src/                    # Source code
│   ├── train_*.py          # Training scripts
│   ├── compare_all_models.py # Model comparison
│   ├── explainability/     # SHAP analysis
│   └── mlflow_utils.py     # MLflow tracking
├── models/                 # Trained models (.joblib)
├── data/                   # Data files
│   ├── raw/                # Original parquet files
│   └── processed/          # Preprocessed CSV files
├── reports/                # Comparison reports & plots
├── Dockerfile              # API container
├── docker-compose.yml      # Multi-service orchestration
└── makefile                # Build commands
```

---

## 🔧 Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn api.main:app --port 8001
```

### Models Not Found

```bash
# Train models first
make train_all

# Check models directory
ls -la models/
```

### Import Errors

```bash
# Reinstall dependencies
make install

# Or
pip install -r requirements.txt --upgrade
```

### Docker Issues

```bash
# Clean Docker
docker-compose down -v
docker system prune -a

# Rebuild
make docker_build
```

---

## 📚 Documentation

- **README.md** - Project overview
- **CLAUDE.md** - Remaining development tasks
- **DEPLOYMENT.md** - Production deployment guide
- **api/README.md** - API documentation

---

## 🎓 Learn More

### SHAP Explanations

```bash
# Run SHAP analysis
python3 -m src.explainability.shap_analysis

# View in dashboard
make app
# → Navigate to "Model Explanation" page
```

### MLflow Experiments

```bash
# Train with tracking
make train_mlflow

# View experiments
make mlflow_ui
# → http://localhost:5000
```

### API Documentation

```bash
# Start API
make api

# Open browser
# → http://localhost:8000/docs (Swagger UI)
# → http://localhost:8000/redoc (ReDoc)
```

---

## ⚡ Performance Tips

1. **Use batch predictions** for multiple samples
2. **Use Docker** for consistent performance
3. **Run with multiple workers** in production: `make api_prod`
4. **Cache model** loaded at startup (already done)
5. **Use Gradient Boosting** for best accuracy

---

## 🎯 Next Steps

1. **Explore Dashboard**: `make app`
2. **Try API**: `make api` → http://localhost:8000/docs
3. **View Experiments**: `make mlflow_ui`
4. **Deploy with Docker**: `make docker_compose_up`
5. **Read CLAUDE.md** for future enhancements

---

**Status**: ✅ Production-Ready

**Questions?** Check documentation or GitHub issues.
