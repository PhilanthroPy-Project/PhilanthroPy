.PHONY: lint typecheck check test doctest coverage ci

lint:
	@echo "==> Linting (flake8 — real defects)..."
	python -m flake8 philanthropy tests examples

typecheck:
	@echo "==> Type checking (mypy)..."
	python -m mypy philanthropy

check:
	@echo "==> Checking for collection errors..."
	@python -m pytest tests/ --collect-only -q || \
		(echo "FATAL: collection errors" && exit 1)
	@echo "OK: no collection errors"

test: check
	@echo "==> Running test suite..."
	python -m pytest tests/ -x --tb=short -q

doctest:
	@echo "==> Running docstring examples..."
	python -m pytest philanthropy --doctest-modules -q --no-cov

coverage: test
	@echo "==> Checking coverage..."
	python -m pytest tests/ --cov=philanthropy --cov-report=term-missing

ci: lint typecheck doctest coverage
	@echo "==> All CI checks passed locally."
