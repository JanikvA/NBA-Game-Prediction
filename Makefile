.PHONY: tests format lint full

full: lint format tests

lint:
	pylama -o pyproject.toml

format:
	black ./
	isort ./
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports **/*.py

tests:
	pytest
