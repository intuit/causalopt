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
	poetry run isort .
	poetry run black .
	poetry run ruff check --fix .

tox:
	poetry run tox lint

setup:
	poetry install --with dev,test

test:
	poetry run pytest
