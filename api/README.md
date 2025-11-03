# Intrusion Detection API

FastAPI-based REST API for real-time network intrusion detection.

## Quick Start

```bash
# Start API server (development mode)
make api

# Or with uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access interactive documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "gradient_boosting",
  "version": "1.0.0",
  "uptime_seconds": 123.45
}
```

### Model Information

```bash
GET /model/info
```

Response:
```json
{
  "name": "gradient_boosting",
  "type": "GradientBoostingClassifier",
  "roc_auc": 0.9787,
  "accuracy": 0.924,
  "f1_score": 0.9287,
  "trained_date": null
}
```

### Single Prediction

```bash
POST /predict
Content-Type: application/json

{
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
  "state": "FIN",
  "sloss": 0,
  "dloss": 0
  // ... other features (see /docs for full schema)
}
```

Response:
```json
{
  "prediction": "attack",
  "probability": 0.95,
  "confidence": "high",
  "model_used": "gradient_boosting",
  "timestamp": "2024-11-02T15:30:00"
}
```

### Batch Prediction

```bash
POST /predict/batch
Content-Type: application/json

{
  "samples": [
    {
      "dur": 0.5,
      "spkts": 20,
      // ... features
    },
    {
      "dur": 0.3,
      "spkts": 15,
      // ... features
    }
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "prediction": "attack",
      "probability": 0.95,
      "confidence": "high",
      "model_used": "gradient_boosting",
      "timestamp": "2024-11-02T15:30:00"
    },
    // ... more predictions
  ],
  "total_samples": 2,
  "attacks_detected": 1,
  "processing_time_ms": 5.23
}
```

## Testing

```bash
# Run API tests
make test_api

# Or manually
python3 test_api.py
```

## Production Deployment

### With uvicorn (multiple workers)

```bash
make api_prod

# Or
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Docker

```bash
docker build -t intrusion-detection-api .
docker run -p 8000:8000 intrusion-detection-api
```

### With Docker Compose

```bash
docker-compose up -d
```

## Performance

- **Single prediction**: ~2-5ms
- **Batch prediction (100 samples)**: ~15-25ms
- **Throughput**: 1000+ requests/second (with 4 workers)

## API Features

✅ **Fast**: < 5ms prediction time
✅ **Scalable**: Multi-worker support
✅ **Documented**: Auto-generated OpenAPI docs
✅ **Validated**: Pydantic models for type safety
✅ **CORS**: Enabled for cross-origin requests
✅ **Error Handling**: Comprehensive error messages
✅ **Batch Processing**: Efficient batch predictions

## Example Usage

### Python

```python
import requests

# Single prediction
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
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### JavaScript

```javascript
const url = 'http://localhost:8000/predict';
const data = {
  dur: 0.5,
  spkts: 20,
  dpkts: 15,
  sbytes: 2048,
  dbytes: 1536,
  rate: 70.0,
  sload: 32768.0,
  dload: 24576.0,
  proto: 'tcp',
  service: 'http',
  state: 'FIN'
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(result => console.log(result));
```

## Security Considerations

- **Authentication**: Not implemented (add API key authentication for production)
- **Rate Limiting**: Not implemented (add for production)
- **HTTPS**: Use reverse proxy (nginx) with SSL in production
- **Input Validation**: Pydantic handles all input validation
- **CORS**: Currently allows all origins (restrict in production)

## Monitoring

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
latency_histogram = Histogram('prediction_latency', 'Prediction latency')
```

## Troubleshooting

**Model not loading**:
- Ensure models are trained: `make train_all`
- Check model path: `models/gradient_boosting.joblib`

**Port already in use**:
```bash
# Use different port
uvicorn api.main:app --port 8001
```

**CORS errors**:
- Check CORS middleware configuration in `api/main.py`

## License

MIT License
