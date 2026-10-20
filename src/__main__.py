import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn
from pydantic import (BaseModel, ConfigDict, Field, TypeAdapter,
                      ValidationError, model_validator)
from llm_sdk.llm_sdk import Small_LLM_Model
from src.json_constrain import run_constrained_json_generation

Type = Literal["str", "string", "int", "integer",
               "float", "number", "bool", "boolean"]
Ident = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
NonBlank = Annotated[str, Field(pattern=r"\S")]  # at least one non-space char


class Strict(BaseModel):
    """Base model that rejects unknown (misspelled) keys."""
    model_config = ConfigDict(extra="forbid")


class ParamSpec(Strict):
    """One parameter of a function definition."""
    type: Type
    description: NonBlank | None = None
    enum: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def enum_only_on_strings(self) -> "ParamSpec":
        if self.enum and self.type not in ("str", "string"):
            raise ValueError("enum is only supported on string parameters")
        return self


class ReturnSpec(Strict):
    """Return type of a function definition."""
    type: Type


class FunctionDef(Strict):
    """One entry of functions_definition.json."""
    name: Ident
    description: NonBlank
    parameters: dict[Ident, ParamSpec]
    returns: ReturnSpec


class PromptEntry(BaseModel):
    """One entry of function_calling_tests.json."""
    prompt: NonBlank


def fail(label: str, exc: ValidationError) -> NoReturn:
    """Exit with a readable list of validation errors."""
    lines = [f"Error: invalid {label}:"]
    for e in exc.errors():
        where = ".".join(map(str, e["loc"])) or "(root)"
        lines.append(f"  - {where}: {e['msg']}")
    sys.exit("\n".join(lines))


def load_json(path: str, what: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        sys.exit(f"Error: cannot read {what} file: {e}")
    except ValueError as e:
        sys.exit(f"Error: {what} file is invalid ({path}): {e}")


def validate_definitions(raw: Any) -> list[dict[str, Any]]:
    try:
        fns = TypeAdapter(list[FunctionDef]).validate_python(raw)
    except ValidationError as e:
        fail("function definitions", e)
    if not fns:
        sys.exit("Error: no function definitions found.")
    names = [f.name for f in fns]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        sys.exit(f"Error: duplicate function names: {dupes}")
    return [f.model_dump(exclude_none=True) for f in fns]


def validate_prompts(raw: Any) -> list[dict[str, Any]]:
    try:
        prompts = TypeAdapter(list[PromptEntry]).validate_python(raw)
    except ValidationError as e:
        fail("input prompts", e)
    return [p.model_dump() for p in prompts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src")
    p.add_argument("--functions_definition",
                   default="data/input/functions_definition.json")
    p.add_argument("--input", default="data/input/function_calling_tests.json")
    p.add_argument("--output", default="data/output/function_calls.json")
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
        print(f"Warning: {len(data) - len(results)} prompt(s) skipped.",
              file=sys.stderr)
    try:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    except OSError as e:
        sys.exit(f"Error: cannot write output file: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
