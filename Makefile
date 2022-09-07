.PHONY: tests format lint full

full: lint format tests

lint:
	mypy nba_game_prediction/ --config-file pyproject.toml
	pylama -o pyproject.toml

format:
	black ./
	isort ./
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports **/*.py
	docformatter --in-place **/*.py

tests:
	pytest -m "not integration"

integration:
	pytest -m "integration" --durations=5
