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
	pytest -m "(not integration) and (not not_with_ga)"

integration:
	pytest -m "integration or not_with_ga" --durations=5
