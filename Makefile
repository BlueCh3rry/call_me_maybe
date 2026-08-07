NAME=main.py

CACHE_DIR ?= /sgoinfre/mmakhmae/tempcache

export POETRY_CACHE_DIR := $(CACHE_DIR)
export PIP_CACHE_DIR := $(CACHE_DIR)/pip

all: ${NAME}

install:
	pip install poetry
	python3 -m poetry install --no-root -C llm_sdk

run:
	uv run python3 ${NAME}

debug:
	python3 -m pdb ${NAME}

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	python3 -m flake8 .
	python3 -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	python3 -m flake8 .
	python3 -m mypy . --strict

.PHONY: install run debug clean lint lint-strict
