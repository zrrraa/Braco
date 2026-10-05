.PHONY: install test validate smoke

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

validate:
	python scripts/validate_config.py --config configs/braco.yaml

smoke: validate test
