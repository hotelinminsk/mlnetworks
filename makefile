.PHONY : prepare train train_all eval compare app clean

prepare:
# 	python -m src.data_ingest
	python3 -m src.preprocess

train:
	python3 -m src.train_iforest

train_sup:
	python3 -m src.train_supervised

train_ensemble:
	python3 -m src.train_ensemble

train_all:
	@echo "Training all models..."
	python3 -m src.train_iforest
	python3 -m src.train_supervised
	python3 -m src.train_ensemble
	@echo "All models trained!"

train_mlflow:
	@echo "Training models with MLflow tracking..."
	python3 -m src.train_with_mlflow
	@echo "Training complete! View results with: mlflow ui"

eval:
	python3 -m src.evaluate

compare:
	python3 -m src.compare_all_models

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

api_prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

test_api:
	python3 test_api.py

clean:
	rm -rf models/*.joblib
	rm -rf reports/*
	@echo "Cleaned models and reports"

install:
	pip install -r requirements.txt

# Docker commands
docker_build:
	docker build -t intrusion-detection-api .
	docker build -f Dockerfile.dashboard -t intrusion-detection-dashboard .

docker_run:
	docker run -d -p 8000:8000 --name ids-api intrusion-detection-api

docker_compose_up:
	docker-compose up -d

docker_compose_down:
	docker-compose down

docker_logs:
	docker-compose logs -f

mlflow_ui:
	mlflow ui

help:
	@echo "Available commands:"
	@echo ""
	@echo "Data & Training:"
	@echo "  make prepare         - Preprocess data"
	@echo "  make train_all       - Train all models"
	@echo "  make train_mlflow    - Train with MLflow tracking"
	@echo "  make eval            - Evaluate models"
	@echo "  make compare         - Compare models and generate report"
	@echo ""
	@echo "Applications:"
	@echo "  make app             - Run Streamlit dashboard"
	@echo "  make api             - Run FastAPI (development)"
	@echo "  make api_prod        - Run FastAPI (production, 4 workers)"
	@echo "  make test_api        - Test API endpoints"
	@echo "  make mlflow_ui       - Launch MLflow UI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker_build    - Build Docker images"
	@echo "  make docker_run      - Run API container"
	@echo "  make docker_compose_up   - Start all services with Docker Compose"
	@echo "  make docker_compose_down - Stop all services"
	@echo "  make docker_logs     - View Docker Compose logs"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           - Clean models and reports"
	@echo "  make install         - Install dependencies"
