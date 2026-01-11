.PHONY: setup install lint format typecheck test pre-commit clean

# Set up development environment
setup: install pre-commit

# Install dependencies
install:
	uv sync --all-groups

# Install pre-commit hooks
pre-commit:
	uv run pre-commit install

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

# Run pre-commit on all files
check:
	uv run pre-commit run --all-files

# Clean up build artifacts
clean:
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
