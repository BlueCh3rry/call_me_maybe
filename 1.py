"""Pydantic models used by the function-calling pipeline."""
import json
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator

# Accepted spellings in functions_definition.json -> canonical type.
TYPE_ALIASES: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "null": "null",
    "none": "null",
}


class ParamSpec(BaseModel):
    """Type of one function parameter."""

    model_config = ConfigDict(extra="allow")

    type: str

    @field_validator("type")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        canonical = TYPE_ALIASES.get(v.strip().lower())
        if canonical is None:
            raise ValueError(f"unsupported parameter type {v!r}")
        return canonical


class FunctionDef(BaseModel):
    """One entry of functions_definition.json."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: dict[str, ParamSpec]
    returns: Optional[dict[str, Any]] = None


class PromptEntry(BaseModel):
    """One entry of function_calling_tests.json."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file (exactly these three keys)."""

    prompt: str
    name: str
    parameters: dict[str, Any]


def build_call(
    prompt: str,
    raw: str,
    functions: list[FunctionDef],
) -> FunctionCall:
    """Parse the generated JSON and cast values to the schema types.

    Raises ValueError if the text does not match the schema.
    """
    try:
        data = json.loads(raw)
        by_name = {f.name: f for f in functions}
        fn = by_name[data["name"]]
        generated = data["parameters"]
        params: dict[str, Any] = {}
        for key, spec in fn.parameters.items():
            val = generated[key]
            if spec.type == "number":
                val = float(val)
            elif spec.type == "integer":
                val = int(val)
            params[key] = val
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"generated output does not match schema: {exc!r}")
    return FunctionCall(prompt=prompt, name=fn.name, parameters=params)