"""Constrained decoding: force the LLM to emit a valid function call.

The target text is always compact JSON (no whitespace):
    {"name":"<fn>","parameters":{"<p1>":<v1>,"<p2>":<v2>}}
At every step an incremental parser says whether a candidate prefix is
"complete", "incomplete" (can still be finished) or "invalid".  Only
tokens that do not make the prefix invalid are allowed.  Text that has
only one possible continuation (keys, punctuation, the end of a unique
function name...) is appended without calling the model at all.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Iterator, Optional

import numpy as np

from src.constrained_decoder import build_token_cache   cv 
from src.models import FunctionDef

MAX_STEPS = 200        # max model calls for one prompt
TOP_K = 64             # candidates checked per step (full scan as fallback)
MAX_STRING_LEN = 160   # a string longer than this must be closed
MAX_DIGITS = 17        # max digits in a number part
COPY_BONUS = 1.5       # logit bonus for text that appears in the prompt
HEX = "0123456789abcdefABCDEF"
ESCAPES = {
    '"': '"', "\\": "\\", "/": "/",
    "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
}


class IncompleteJSON(Exception):
    """The text is a valid prefix. `forced` is text that must follow."""

    def __init__(self, forced: str = "") -> None:
        super().__init__()
        self.forced = forced


class InvalidJSON(Exception):
    """The text cannot be extended into a valid function call."""


# --------------------------------------------------------------------------
# Incremental parser
# --------------------------------------------------------------------------

def expect(text: str, pos: int, expected: str) -> int:
    end = pos + len(expected)
    actual = text[pos:end]
    if actual == expected:
        return end
    if expected.startswith(actual):
        raise IncompleteJSON(expected[len(actual):])
    raise InvalidJSON(f"expected {expected!r}, got {actual!r}")


def match(text: str, pos: int, options: list[str]) -> int:
    rest = text[pos:]
    for option in options:
        if rest.startswith(option):
            return pos + len(option)
    if any(option.startswith(rest) for option in options):
        raise IncompleteJSON()
    raise InvalidJSON(f"expected one of {options}, got {rest!r}")


def forced_for_enum(cur: str, allowed: Optional[list[str]]) -> str:
    """Text that is certain to follow `cur` inside an enum string."""
    if allowed is None:
        return ""
    cands = [a for a in allowed if a.startswith(cur)]
    if not cands:
        return ""
    if len(cands) == 1:
        body = cands[0][len(cur):]
        tail = '"'
    else:
        body = os.path.commonprefix(cands)[len(cur):]
        tail = ""
    if any(ch in '"\\' or ord(ch) < 32 for ch in body):
        return ""
    return body + tail


def string(
    text: str, pos: int, allowed: Optional[list[str]] = None
) -> tuple[int, str]:
    """Parse a JSON string. `allowed` restricts it to an enum."""
    if pos >= len(text):
        raise IncompleteJSON('"')
    if text[pos] != '"':
        raise InvalidJSON(f"expected '\"' at {pos}")
    pos += 1
    out: list[str] = []
    while pos < len(text):
        c = text[pos]
        pos += 1
        if c == '"':
            val = "".join(out)
            if allowed is not None and val not in allowed:
                raise InvalidJSON(f"{val!r} not allowed")
            return pos, val
        if c == "\\":
            if allowed is not None:
                raise InvalidJSON("escape inside enum string")
            if pos >= len(text):
                raise IncompleteJSON()
            c = text[pos]
            pos += 1
            if c in ESCAPES:
                out.append(ESCAPES[c])
            elif c == "u":
                if pos + 4 > len(text):
                    raise IncompleteJSON()
                digits = text[pos:pos + 4]
                if not all(x in HEX for x in digits):
                    raise InvalidJSON("bad unicode escape")
                out.append(chr(int(digits, 16)))
                pos += 4
            else:
                raise InvalidJSON(f"bad escape \\{c}")
        elif ord(c) < 0x20:
            raise InvalidJSON("raw control character in string")
        else:
            out.append(c)
        cur = "".join(out)
        if allowed is not None and not any(a.startswith(cur) for a in allowed):
            raise InvalidJSON(f"{cur!r} cannot continue")
        if len(cur) > MAX_STRING_LEN:
            raise InvalidJSON("string too long")
    raise IncompleteJSON(forced_for_enum("".join(out), allowed))


def _digits(text: str, pos: int) -> int:
    """Consume at least one digit (<= MAX_DIGITS). Returns the new pos."""
    start = pos
    while pos < len(text) and text[pos].isdigit():
        pos += 1
        if pos - start > MAX_DIGITS:
            raise InvalidJSON("number too long")
    if pos == start:
        if pos >= len(text):
            raise IncompleteJSON()
        raise InvalidJSON("digit expected")
    return pos


def number(text: str, pos: int, integer: bool = False) -> int:
    if pos < len(text) and text[pos] == "-":
        pos += 1
    first = pos
    pos = _digits(text, pos)
    if pos - first > 1 and text[first] == "0":
        raise InvalidJSON("leading zero")
    if not integer:
        if pos < len(text) and text[pos] == ".":
            pos = _digits(text, pos + 1)
        if pos < len(text) and text[pos] in "eE":
            pos += 1
            if pos < len(text) and text[pos] in "+-":
                pos += 1
            pos = _digits(text, pos)
    if pos >= len(text):
        # more digits could still come
        raise IncompleteJSON()
    return pos


def value(text: str, pos: int, typ: str) -> int:
    if typ == "string":
        return string(text, pos)[0]
    if pos >= len(text):
        raise IncompleteJSON()
    if typ == "number":
        return number(text, pos)
    if typ == "integer":
        return number(text, pos, integer=True)
    if typ == "boolean":
        return match(text, pos, ["true", "false"])
    if typ == "null":
        return match(text, pos, ["null"])
    raise InvalidJSON(f"unsupported type: {typ}")


def parse_generation(text: str, functions: list[FunctionDef]) -> int:
    names = [f.name for f in functions]
    pos = expect(text, 0, '{"name":')
    pos, name = string(text, pos, allowed=names)
    fn = next(f for f in functions if f.name == name)
    pos = expect(text, pos, ',"parameters":{')
    keys = list(fn.parameters)
    for i, key in enumerate(keys):
        pos = expect(text, pos, json.dumps(key) + ":")
        pos = value(text, pos, fn.parameters[key].type)
        if i < len(keys) - 1:
            pos = expect(text, pos, ",")
    pos = expect(text, pos, "}}")
    if pos != len(text):
        raise InvalidJSON("trailing characters")
    return pos


def classify(text: str, functions: list[FunctionDef]) -> tuple[str, str]:
    """Return (status, forced_text)."""
    try:
        parse_generation(text, functions)
        return "complete", ""
    except IncompleteJSON as exc:
        return "incomplete", exc.forced
    except InvalidJSON:
        return "invalid", ""


def advance(text: str, functions: list[FunctionDef]) -> tuple[str, str, str]:
    """Append all forced text. Returns (text, added_text, status)."""
    added = ""
    while True:
        status, forced = classify(text, functions)
        if status != "incomplete" or not forced:
            return text, added, status
        text += forced
        added += forced


# --------------------------------------------------------------------------
# Token selection
# --------------------------------------------------------------------------

def open_value_string(text: str) -> Optional[str]:
    """Content so far if `text` ends inside a *parameter value* string."""
    in_str = False
    escaped = False
    start = 0
    out: list[str] = []
    for i, c in enumerate(text):
        if in_str:
            if escaped:
                out.append(c)
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
            else:
                out.append(c)
        elif c == '"':
            in_str = True
            start = i
            out = []
    if not in_str or start == 0 or text[start - 1] != ":":
        return None
    if '"parameters"' not in text[:start]:
        return None
    return "".join(out)


def copy_bonus(text: str, token: str, prompt: str) -> float:
    """Small bonus when a string value stays a piece of the user prompt.

    Generic (no parameter names): most string arguments are copied from
    the request, others (regex, replacement...) are simply not boosted.
    """
    content = open_value_string(text + token)
    if content and content in prompt:
        return COPY_BONUS
    return 0.0


def token_ok(
    tid: int,
    text: str,
    id_to_str: dict[int, str],
    functions: list[FunctionDef],
) -> bool:
    token = id_to_str.get(tid)
    if not token:
        return False
    return classify(text + token, functions)[0] != "invalid"


def pick_token(
    logits: Any,
    text: str,
    id_to_str: dict[int, str],
    functions: list[FunctionDef],
    prompt: str,
) -> Optional[int]:
    """Best valid token (greedy), checking candidates in logit order."""
    arr = np.asarray(logits, dtype=np.float64)
    k = min(TOP_K, arr.shape[0] - 1)
    top = np.argpartition(-arr, k)[:k]
    top = top[np.argsort(-arr[top])]
    in_value = open_value_string(text) is not None
    max_bonus = COPY_BONUS if in_value else 0.0
    best_id: Optional[int] = None
    best_score = float("-inf")
    for raw_id in top:
        tid = int(raw_id)
        score = float(arr[tid])
        if best_id is not None and score + max_bonus <= best_score:
            break
        if not token_ok(tid, text, id_to_str, functions):
            continue
        score += copy_bonus(text, id_to_str[tid], prompt)
        if score > best_score:
            best_id, best_score = tid, score
    if best_id is not None:
        return best_id
    for raw_id in np.argsort(-arr):          # rare: nothing valid in top-k
        tid = int(raw_id)
        if token_ok(tid, text, id_to_str, functions):
            return tid
    return None


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

# Few-shot examples. An example is only used when the function exists in
# the loaded definitions with exactly the same parameter names.
FEW_SHOT: list[tuple[str, str, dict[str, Any]]] = [
    ("Add 4 and 7", "fn_add_numbers", {"a": 4, "b": 7}),
    ("Say hello to Anna", "fn_greet", {"name": "Anna"}),
    ("Reverse the string 'abc'", "fn_reverse_string", {"s": "abc"}),
    ("What is the square root of 81?", "fn_get_square_root", {"a": 81}),
    (
        "Replace all digits in 'a1b22' with X",
        "fn_substitute_string_with_regex",
        {"source_string": "a1b22", "regex": "\\d+", "replacement": "X"},
    ),
    (
        "Replace all vowels in 'banana' with #",
        "fn_substitute_string_with_regex",
        {
            "source_string": "banana",
            "regex": "[aeiouAEIOU]",
            "replacement": "#",
        },
    ),
    (
        "Replace the word 'red' with 'blue' in 'red car, red bike'",
        "fn_substitute_string_with_regex",
        {
            "source_string": "red car, red bike",
            "regex": "red",
            "replacement": "blue",
        },
    ),
]


def usable_examples(
    functions: list[FunctionDef],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    by_name = {f.name: f for f in functions}
    for question, name, params in FEW_SHOT:
        fn = by_name.get(name)
        if fn is not None and set(params) == set(fn.parameters):
            yield question, name, {k: params[k] for k in fn.parameters}


def compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def describe(fn: FunctionDef) -> str:
    params = ", ".join(f"{k}: {p.type}" for k, p in fn.parameters.items())
    return f"- {fn.name}({params}): {fn.description}"


def build_prompt(user_prompt: str, functions: list[FunctionDef]) -> str:
    system = (
        "You turn a user request into exactly one function call.\n"
        "Answer with compact JSON only, no spaces, no explanation: "
        '{"name":"<function>","parameters":{...}}\n'
        "Pick the function whose description matches the request. "
        "Copy text values exactly as written in the request, without "
        "the quotes around them. For a regex parameter write the "
        "shortest pattern that matches only the text to replace; never "
        "wrap it in .* unless asked.\n"
        "Functions:\n" + "\n".join(describe(f) for f in functions)
    )
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for question, name, params in usable_examples(functions):
        answer = compact({"name": name, "parameters": params})
        parts.append(f"<|im_start|>user\n{question}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{answer}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>\n")
    # empty think block = Qwen3 non-thinking mode
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate_call(
    model: Any,
    id_to_str: dict[int, str],
    functions: list[FunctionDef],
    user_prompt: str,
) -> Optional[str]:
    """Return the generated JSON text, or None if generation failed."""
    ids: list[int] = model.encode(
        build_prompt(user_prompt, functions)
    ).tolist()[0]
    text = ""
    steps = 0
    while True:
        text, added, status = advance(text, functions)
        if added:
            ids += model.encode(added).tolist()[0]
        if status == "complete":
            return text
        if status == "invalid" or steps >= MAX_STEPS:
            return None
        logits = model.get_logits_from_input_ids(ids)
        tid = pick_token(logits, text, id_to_str, functions, user_prompt)
        if tid is None:
            return None
        ids.append(tid)
        text += id_to_str[tid]
        steps += 1


def run_constrained_json_generation(
    prompts: list[str],
    model: Any,
    vocab_size: int,
    functions: list[FunctionDef],
) -> list[Optional[str]]:
    """Generate one JSON function call per prompt (None on failure)."""
    id_to_str = build_token_cache(model, vocab_size)
    results: list[Optional[str]] = []
    total = len(prompts)
    for i, user_prompt in enumerate(prompts, start=1):
        started = time.perf_counter()
        try:
            text = generate_call(model, id_to_str, functions, user_prompt)
        except Exception as exc:  # keep going with the other prompts
            print(f"[{i}/{total}] generation error: {exc!r}",
                  file=sys.stderr)
            text = None
        elapsed = time.perf_counter() - started
        print(f"[{i}/{total}] {user_prompt!r} -> {text} ({elapsed:.1f}s)")
        results.append(text)
    return results