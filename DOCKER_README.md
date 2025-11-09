# Docker Setup Guide

## 🐳 Docker Yapılandırması

### Gereksinimler
- Docker 20.10+
- Docker Compose 2.0+

### Hızlı Başlangıç

```bash
# Tüm servisleri başlat
docker-compose up -d

# Sadece dashboard
docker-compose up dashboard

# Sadece API
docker-compose up api
```

### Servisler

#### 1. **API** (Port 8000)
- FastAPI intrusion detection API
- Health check: `http://localhost:8000/health`
- Docs: `http://localhost:8000/docs`

#### 2. **Dashboard** (Port 8501)
- Streamlit dashboard
- Health check: `http://localhost:8501/_stcore/health`
- URL: `http://localhost:8501`

#### 3. **MLflow** (Port 5000)
- Model tracking server
- URL: `http://localhost:5000`

#### 4. **Prometheus** (Port 9090) - Optional
- Monitoring metrics
- Profile: `docker-compose --profile monitoring up prometheus`

#### 5. **Grafana** (Port 3000) - Optional
- Monitoring dashboards
- Profile: `docker-compose --profile monitoring up grafana`

## 📦 Build

### Dashboard
```bash
docker build -f Dockerfile.dashboard -t ids-dashboard .
```

### API
```bash
docker build -f Dockerfile -t ids-api .
```

## 🔧 Yapılandırma

### Environment Variables

**API:**
- `MODEL_NAME`: Default model (default: `gradient_boosting`)
- `PYTHONUNBUFFERED=1`: Python output buffering

**Dashboard:**
- `PYTHONUNBUFFERED=1`: Python output buffering

### Volumes

- `./models` → Model dosyaları
- `./data` → Dataset dosyaları
- `./processed` → İşlenmiş veriler
- `./reports` → Raporlar

## 🚀 Kullanım

### Tüm Servisleri Başlat
```bash
docker-compose up -d
```

### Logları İzle
```bash
docker-compose logs -f dashboard
docker-compose logs -f api
```

### Servisleri Durdur
```bash
docker-compose down
```

### Servisleri Yeniden Başlat
```bash
docker-compose restart dashboard
docker-compose restart api
```

### Container'a Bağlan
```bash
# Dashboard
docker exec -it intrusion-detection-dashboard bash

# API
docker exec -it intrusion-detection-api bash
```

## 🏥 Health Checks

### API Health
```bash
curl http://localhost:8000/health
```

### Dashboard Health
```bash
curl http://localhost:8501/_stcore/health
```

## 📝 Notlar

- **Models**: Model dosyaları volume olarak mount edilir
- **Data**: Data dosyaları volume olarak mount edilir
- **Health Checks**: Her servis için health check tanımlı
- **Non-root User**: Güvenlik için non-root user kullanılıyor
- **Multi-stage Build**: API için optimized multi-stage build

## 🔍 Troubleshooting

### Port Already in Use
```bash
# Port'u kullanan process'i bul
lsof -i :8501
lsof -i :8000

# Process'i durdur veya docker-compose.yml'de port'u değiştir
```

### Model Files Not Found
```bash
# Model dosyalarını kontrol et
ls -la models/

# Model eğit
./venv/bin/python -m src.train_all_models
```

### Container Won't Start
```bash
# Logları kontrol et
docker-compose logs dashboard
docker-compose logs api

# Container'ı rebuild et
docker-compose build --no-cache dashboard
docker-compose up dashboard
```

## 📊 Monitoring (Optional)

Monitoring servislerini başlatmak için:
```bash
docker-compose --profile monitoring up -d
```

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

