.PHONY: install setup run clean test lint

# Python environment setup
install:
	pip install -e .
	pip install -r requirements.txt

setup: install
	python -m camel_tools.data_downloader light
	# Add any other setup steps here

# Running the application
run:
	python -m lemmatizer.web.app

# Development tools
test:
	pytest tests/

lint:
	flake8 src/ lemmatizer/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -exec rm -f {} +

