# Makefile — Common project tasks
# Usage: make <target>
# Requires: Python 3.9+, pip

.PHONY: help install setup download test run dashboard clean

help:
	@echo "Real-Time Object Detection — Available targets:"
	@echo ""
	@echo "  make install       Install all Python dependencies"
	@echo "  make setup         Install + download sample videos"
	@echo "  make download      Download sample video only"
	@echo "  make test          Run the full test suite"
	@echo "  make run           Run the full detection pipeline"
	@echo "  make dashboard     Launch the Streamlit dashboard"
	@echo "  make clean         Remove generated outputs (keeps models)"
	@echo ""

install:
	pip install -r requirements.txt

setup: install download
	@echo "Setup complete. Run 'make run' to start the pipeline."

download:
	python scripts/download_samples.py

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=detection_system --cov-report=term-missing

run:
	python scripts/run_all.py

run-quick:
	python scripts/run_all.py --max_frames 100 --skip_benchmark

dashboard:
	streamlit run app/streamlit_app.py

clean:
	@echo "Removing generated outputs..."
	rm -f data/processed/*.csv data/processed/*.json data/processed/*.mp4
	rm -f data/interim/*.jpg data/interim/*.png
	rm -f reports/figures/*.png reports/figures/*.pdf
	rm -f reports/tables/*.csv reports/tables/*.tex
	@echo "Clean complete. Models preserved in models/."

clean-all: clean
	@echo "Also removing downloaded model weights..."
	rm -f models/*.pt
