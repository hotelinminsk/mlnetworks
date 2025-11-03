# Deployment Guide

Complete guide for deploying the Intrusion Detection System to production.

## Quick Start with Docker

### 1. Build and Run

```bash
# Build Docker image
docker build -t intrusion-detection-api .

# Run container
docker run -d -p 8000:8000 --name ids-api intrusion-detection-api

# Test
curl http://localhost:8000/health
```

### 2. Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. With Monitoring (Optional)

```bash
# Start with Prometheus and Grafana
docker-compose --profile monitoring up -d

# Access:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

## Manual Deployment

### Prerequisites

- Python 3.9+
- pip
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mlnetworks.git
cd mlnetworks

# Install dependencies
pip install -r requirements.txt

# Train models (if not already trained)
make train_all

# Run API
make api
```

## Production Deployment Options

### Option 1: Cloud VM (AWS EC2, Google Cloud, etc.)

#### 1. Setup VM

```bash
# SSH into VM
ssh user@your-server-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 2. Deploy

```bash
# Clone repository
git clone https://github.com/yourusername/mlnetworks.git
cd mlnetworks

# Start services
docker-compose up -d

# Setup reverse proxy (nginx)
sudo apt install nginx
sudo nano /etc/nginx/sites-available/intrusion-detection
```

Nginx config:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /dashboard {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 3. Enable HTTPS

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### Option 2: Kubernetes

#### 1. Create Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: intrusion-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: intrusion-detection
  template:
    metadata:
      labels:
        app: intrusion-detection
    spec:
      containers:
      - name: api
        image: intrusion-detection-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: intrusion-detection-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: intrusion-detection
```

#### 2. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get services
```

### Option 3: Serverless (AWS Lambda + API Gateway)

1. Use AWS Lambda with FastAPI
2. Deploy with Mangum adapter
3. Setup API Gateway

### Option 4: Platform as a Service

#### Heroku

```bash
# Create Procfile
echo "web: uvicorn api.main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### Google Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/intrusion-detection

# Deploy
gcloud run deploy intrusion-detection \
  --image gcr.io/PROJECT_ID/intrusion-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Environment Variables

```bash
# API Configuration
MODEL_NAME=gradient_boosting
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# MLflow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=intrusion-detection

# Security
API_KEY=your-secret-key  # Implement in api/main.py
CORS_ORIGINS=https://your-domain.com

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=info
```

## Scaling

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up --scale api=4

# Kubernetes
kubectl scale deployment intrusion-detection-api --replicas=10
```

### Load Balancing

Use nginx or cloud load balancer:

```nginx
upstream api_backend {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    location / {
        proxy_pass http://api_backend;
    }
}
```

## Monitoring & Logging

### Application Logs

```bash
# Docker
docker-compose logs -f api

# Kubernetes
kubectl logs -f deployment/intrusion-detection-api

# Save logs
docker-compose logs > logs.txt
```

### Metrics

Access Prometheus at http://localhost:9090
Access Grafana at http://localhost:3000

## Security Checklist

- [ ] Enable HTTPS/TLS
- [ ] Implement API key authentication
- [ ] Add rate limiting
- [ ] Configure CORS properly
- [ ] Use secrets management (AWS Secrets Manager, etc.)
- [ ] Enable firewall rules
- [ ] Regular security updates
- [ ] Implement request validation
- [ ] Setup monitoring and alerts
- [ ] Backup models and data regularly

## Performance Optimization

### 1. Model Caching

Models are loaded once at startup and cached in memory.

### 2. Batch Predictions

Use `/predict/batch` endpoint for multiple predictions.

### 3. Connection Pooling

Configure uvicorn workers:

```bash
uvicorn api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### 4. Response Compression

Enable gzip compression in nginx:

```nginx
gzip on;
gzip_types application/json;
```

## Troubleshooting

### API Not Starting

```bash
# Check logs
docker-compose logs api

# Verify model files exist
ls models/

# Check ports
lsof -i :8000
```

### High Memory Usage

```bash
# Monitor memory
docker stats

# Reduce workers
uvicorn api.main:app --workers 2
```

### Slow Predictions

```bash
# Check model size
du -sh models/*

# Use batch predictions
# Optimize preprocessing
```

## Backup & Recovery

### Backup Models

```bash
# Backup models directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Upload to cloud storage
aws s3 cp models_backup_*.tar.gz s3://your-bucket/backups/
```

### Restore Models

```bash
# Download from cloud
aws s3 cp s3://your-bucket/backups/models_backup_20241102.tar.gz .

# Extract
tar -xzf models_backup_20241102.tar.gz
```

## Updates & Rollbacks

### Deploy New Version

```bash
# Pull latest code
git pull origin main

# Rebuild
docker-compose build

# Rolling update
docker-compose up -d --no-deps --build api
```

### Rollback

```bash
# Docker
docker tag intrusion-detection-api:latest intrusion-detection-api:backup
docker pull intrusion-detection-api:previous-version
docker-compose up -d

# Kubernetes
kubectl rollout undo deployment/intrusion-detection-api
```

## Maintenance

### Regular Tasks

- **Daily**: Check logs and metrics
- **Weekly**: Review API usage and performance
- **Monthly**: Update dependencies, retrain models
- **Quarterly**: Security audit, load testing

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_sample.json
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/mlnetworks/issues
- Email: support@yourcompany.com

## License

MIT License
