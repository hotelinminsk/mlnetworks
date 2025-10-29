.PHONY : prepare train eval app

prepare:
# 	python -m src.data_ingest
	python -m src.preprocess

train:
	python -m src.train_iforest

train_sup:
	python -m src.train_supervised

eval:
	python -m src.evaluate

app:
	streamlit run app/streamlit_app.py
