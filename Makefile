NAME=main.py

CACHE_DIR ?= /sgoinfre/mmakhmae/tempcache

export UV_CACHE_DIR := $(CACHE_DIR)/uv
export UV_LINK_MODE := copy
export HF_HOME := $(CACHE_DIR)/huggingface

all: install run

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json

debug:
	uv run python -m pdb -m src

clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" \) \
		-not -path "./.venv/*" -prune -exec rm -rf {} +

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

.PHONY: all install run debug clean lint lint-strict