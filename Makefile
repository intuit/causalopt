VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +
	@find . -type f -name "*.xml*" -exec rm -rf {} +
	@find . -type f -name "*.coverage*" -exec rm -rf {} +
	@find . -type f -name "*.isorted" -exec rm -rf {} +
	@find . -type f -name "*.htmlcov_*" -exec rm -rf {} +

lint:
	$(VENV)/bin/isort .
	$(VENV)/bin/black .
	$(VENV)/bin/ruff check --fix .

tox:
	poetry run tox lint

setup:
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet isort "black[jupyter]" ruff pytest pytest-cov pytest-xdist
	$(PIP) install --quiet -e .

test:
	$(VENV)/bin/pytest
