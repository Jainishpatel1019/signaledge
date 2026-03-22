.PHONY: install test run process docker-build docker-run

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short

run:
	streamlit run app.py

process:
	python -c "from src.pipeline.processor import CompanyProcessor; CompanyProcessor().process('AAPL')"

docker-build:
	docker build -t signaledge:v1 .

docker-run:
	docker run -p 8501:8501 signaledge:v1
