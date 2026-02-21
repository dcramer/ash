.PHONY: setup install dotagents-install lint format typecheck test hooks clean

# Set up development environment
setup: install hooks

# Install dependencies
install:
	uv sync --all-groups
	$(MAKE) dotagents-install

# Install/update dotagents configuration
dotagents-install:
	npx --yes @sentry/dotagents install

# Install prek hooks
hooks:
	prek install

# Run all linters and formatters
lint:
	uv run ruff check --fix .
	uv run ruff format .

# Format only (no lint fixes)
format:
	uv run ruff format .

# Type check
typecheck:
	uv run ty check

# Run tests
test:
	uv run pytest tests/ -v

# Run prek on all files
check:
	prek run --all-files

# Clean up build artifacts
clean:
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
