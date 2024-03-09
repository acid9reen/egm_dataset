# Dev setup
.PHONY: deps
deps:
	# Note: to run you'll need uv installed
	uv pip compile pyproject.toml -o requirements/requirements.txt
	uv pip compile pyproject.toml --extra=dev --extra=test -o requirements/dev-requirements.txt
	uv pip sync requirements/requirements.txt requirements/dev-requirements.txt
	pip install -e . --no-deps

.PHONY: lint
lint:
	-ruff check
	-mypy src

.PHONY: test
test:
	pytest

.PHONY: format
format:
	ruff format src
	ruff check --select "I" --fix
