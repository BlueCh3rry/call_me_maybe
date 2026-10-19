import argparse
import json
import sys
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
from pathlib import Path
from src.json_constrain import run_constrained_json_generation

VALID_TYPES = {"str", "string", "int", "integer",
               "float", "number", "bool", "boolean"}


def load_json(path: str, what: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: {what} file not found: {path}")
    except IsADirectoryError:
        sys.exit(f"Error: {what} path is a directory: {path}")
    except PermissionError:
        sys.exit(f"Error: no permission to read {what} file: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error: {what} file is not valid JSON ({path}): {e}")
    except UnicodeDecodeError:
        sys.exit(f"Error: {what} file is not valid UTF-8: {path}")


def validate_definitions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        sys.exit("Error: function definitions must be a JSON list.")
    good, seen = [], set()
    for idx, fn in enumerate(raw):
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            print(f"Warning: definition #{idx} (needs a string 'name').",
                  file=sys.stderr)
            sys.exit(1)
        if fn["name"] in seen:
            print(f"Warning: duplicate definition '{fn['name']}' skipped.",
                  file=sys.stderr)
            sys.exit(1)
        params = fn.get("parameters", {})
        if not isinstance(params, dict) or not all(
            isinstance(p, dict) and p.get("type") in VALID_TYPES
            for p in params.values()
        ):
            print(f"Warning: definition '{fn['name']}' skipped "
                  "(bad 'parameters').", file=sys.stderr)
            sys.exit(1)
        seen.add(fn["name"])
        good.append({**fn, "parameters": params})
    if not good:
        sys.exit("Error: no usable function definitions.")
    return good


def validate_prompts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        sys.exit("Error: input file must be a JSON list of {\"prompt\": ...}.")
    good = []
    for idx, entry in enumerate(raw):
        if (isinstance(entry, dict) and isinstance(entry.get("prompt"), str)
                and entry["prompt"].strip()):
            good.append(entry)
        else:
            print(f"Warning: prompt #{idx} (needs non-empty 'prompt').",
                  file=sys.stderr)
            sys.exit(1)
    return good


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src")
    p.add_argument("--functions_definition",
                   default="data/input/functions_definition.json")
    p.add_argument("--input", default="data/input/function_calling_tests.json")
    p.add_argument("--output",
                   default="data/output/function_calls.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    schemas = validate_definitions(
        load_json(args.functions_definition, "functions definition"))
    data = validate_prompts(load_json(args.input, "input"))
    if not data:
        sys.exit("Error: no valid prompts to process.")
    try:
        model = Small_LLM_Model()
    except Exception as e:
        sys.exit(f"Error: could not load model: {e}")
    calls = run_constrained_json_generation(data, model, schemas)

    results = [c for c in calls if c is not None]
    if len(results) < len(data):
        print(f"Warning: {len(data) - len(results)} prompt(s) failed.",
              file=sys.stderr)
    try:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except OSError as e:
        sys.exit(f"Error: cannot write output file: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
