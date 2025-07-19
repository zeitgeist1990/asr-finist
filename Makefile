CODE = tone

.PHONY: install
install:
	poetry install -E demo

.PHONY: lint
lint:
	# check package version
	python -c "import tone as a; exit(a.__version__ != \"`poetry version -s`\")"
	poetry check --lock
	# python linters
	ruff check $(CODE)
	ruff format --check $(CODE)

.PHONY: format
format:
	ruff format $(CODE)
	ruff check --fix $(CODE)

.PHONY: up_dev
up_dev:
	uvicorn --host 0.0.0.0 --port 8080 tone.demo.website:app --reload
