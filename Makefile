.PHONY: lint check test coverage ci

lint:
	@echo "==> Linting (flake8 — real defects)..."
	python -m flake8 philanthropy tests examples

check:
	@echo "==> Checking for collection errors..."
	@python -m pytest tests/ --collect-only -q || \
		(echo "FATAL: collection errors" && exit 1)
	@echo "OK: no collection errors"

test: check
	@echo "==> Running test suite..."
	python -m pytest tests/ -x --tb=short -q

coverage: test
	@echo "==> Checking coverage..."
	python -m pytest tests/ --cov=philanthropy --cov-fail-under=85 --cov-report=term-missing

ci: lint coverage
	@echo "==> All CI checks passed locally."
