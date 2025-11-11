PYTHON ?= python3

.PHONY: install app clean docker_build docker_run help

install:
	pip install -r requirements.txt

app:
	streamlit run app/ids_dashboard_final.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .streamlit

docker_build:
	docker build -t ids-dashboard .

docker_run:
	docker run --rm -p 8501:8501 ids-dashboard

help:
	@echo "Targets: install | app | clean | docker_build | docker_run"
