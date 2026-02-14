.PHONY: install dev test lint typecheck clean run help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -e .

dev:  ## Install with development dependencies
	pip install -e ".[dev]"
	pre-commit install || true

test:  ## Run the test suite
	pytest -v --tb=short

test-fast:  ## Run tests excluding slow / network tests
	pytest -v --tb=short -m "not slow and not integration"

test-cov:  ## Run tests with coverage report
	pytest --cov=fusion_oncology --cov-report=term-missing --cov-report=html

lint:  ## Lint source code with ruff
	ruff check src/ tests/

lint-fix:  ## Auto-fix lint issues
	ruff check --fix src/ tests/

typecheck:  ## Run static type checking
	mypy src/fusion_oncology/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run:  ## Run the full fusion pipeline
	fusion-oncology run

run-quick:  ## Run pipeline with reduced iterations (faster)
	fusion-oncology run --fuzz-iterations 5 --top-k 3
