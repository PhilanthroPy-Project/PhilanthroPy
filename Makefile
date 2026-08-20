.PHONY: lint typecheck check test doctest coverage riskcov ci

# Single source of truth for the risk-tier coverage floor. CI and CONTRIBUTING.md
# both call `make riskcov` so the include list cannot drift between them.
RISK_TIER := philanthropy/preprocessing/*,philanthropy/models/*,philanthropy/ingest/*,philanthropy/cli.py,philanthropy/utils/_persistence.py
RISK_FLOOR := 93

lint:
	@echo "==> Linting (flake8, real defects)..."
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

coverage: check
	@echo "==> Checking coverage..."
	python -m pytest tests/ --cov=philanthropy --cov-report=term-missing

riskcov:
	@echo "==> Checking risk-tier coverage floor ($(RISK_FLOOR)%)..."
	python -m coverage report --include='$(RISK_TIER)' --fail-under=$(RISK_FLOOR)

ci: lint typecheck doctest coverage
	@echo "==> All CI checks passed locally."
