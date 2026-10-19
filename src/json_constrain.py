import json
import re
import sys
import typing
import numpy as np
from typing import Any
from llm_sdk.llm_sdk import Small_LLM_Model as SLM
from pydantic import BaseModel, ConfigDict


DEBUG = False
MAX_STRING_TOKENS = 60
MAX_NUMBER_CHARS = 20

INT_TYPES = {"int", "integer"}
NUM_TYPES = {"float", "number"}
BOOL_TYPES = {"bool", "boolean"}

STR_BODY = re.compile(r'(?:[^"\\\x00-\x1f]|\\.)*\\?')
STR_DONE = re.compile(r'(?:[^"\\\x00-\x1f]|\\.)*')
NUM_PREFIX = re.compile(r"-?(?:\d+(?:\.\d*)?)?")
INT_PREFIX = re.compile(r"-?\d*")
NUM_FULL = re.compile(r"-?\d+(?:\.\d+)?")
INT_FULL = re.compile(r"-?\d+")
ESC = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\/nrt])')
SIMPLE_ESC = {'"': '"', "\\": "\\", "/": "/", "n": "\n", "r": "\r", "t": "\t"}


def unescape(raw: str) -> str:
    def sub(m: re.Match[str]) -> str:
        g = m.group(1)
        return chr(int(g[1:], 16)) if g[0] == "u" else SIMPLE_ESC[g]
    return ESC.sub(sub, raw)


def byte_decoder() -> dict[str, int]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_vocabulary(model: SLM) -> list[typing.Optional[str]]:
    """tokens[id] = real text of the token, None if it can't be text."""
    with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
        raw = json.load(f)
    dec = byte_decoder()
    tokens: list[typing.Optional[str]] = [None] * (max(raw.values()) + 1)
    for tok, tid in raw.items():
        try:
            tokens[tid] = bytes(dec[c] for c in tok).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            pass
    return tokens


def build_prompt(user_prompt: str, schemas: list[dict[str, Any]]) -> str:
    funcs = "\n".join(
        f"- {s['name']}: {s.get('description', '')} "
        f"Parameters: {json.dumps(s.get('parameters', {}))}"
        for s in schemas
    )
    return (
        "<|im_start|>system\n"
        "You convert a user request into a single function call.\n"
        f"Available functions:\n{funcs}\n\n"
        "Rules:\n"
        "- Pick the function whose description best matches the request.\n"
        "- Do not add any leading space or trailing spaces unless asked.\n"
        "- Copy string arguments exactly from the request, without the "
        "surrounding quotes.\n"
        "- Numbers are plain digits.\n"
        "- For regex parameters, write a regex that matches what the user "
        "describes.\n"
        "Respond with JSON only: "
        '{"name": "<function name>", "parameters": {<param>: <value>, ...}}'
        "\n/no_think<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class Decoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: SLM
    schemas: list[dict[str, Any]]
    tokens: list[str | None] = []
    by_text: dict[str, list[int]] = {}
    number_ids: list[int] = []
    end_ids: list[int] = []
    end_set: set[int] = set()

    def model_post_init(self, __context: Any) -> None:
        self.tokens = load_vocabulary(self.model)
        for i, t in enumerate(self.tokens):
            if not t:
                continue
            self.by_text.setdefault(t, []).append(i)
            if all(c in "0123456789.-" for c in t):
                self.number_ids.append(i)
            if t[0] in ",}":
                self.end_ids.append(i)
        self.end_set = set(self.end_ids)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [int(i) for i in self.model.encode(text)[0].tolist()]

    def logits(self, ids: list[int]) -> np.typing.NDArray[np.float64]:
        return np.asarray(
            self.model.get_logits_from_input_ids(ids), dtype=np.float64)

    def pick(self, ids: list[int], allowed: list[int]) -> int:
        """Mask every token not in `allowed` with -inf, then argmax."""
        if not allowed:
            raise ValueError("no valid token at this step")
        lg = self.logits(ids)
        allowed = [i for i in allowed if i < len(lg)]
        masked = np.full(len(lg), -np.inf)
        masked[allowed] = lg[allowed]
        return int(np.argmax(masked))

    def choose(self, ids: list[int], options: list[str]) -> str:
        """Model picks one option. Auto-completes when only one is left."""
        text: str = ""
        while True:
            left = [o for o in options if o.startswith(text)]
            if not left:
                raise ValueError("no option matches")
            if len(left) == 1:
                ids.extend(self.encode(left[0][len(text):]))
                return left[0]
            allowed: list[int] = []
            for o in left:
                rest = o[len(text):]
                for n in range(1, len(rest) + 1):
                    allowed.extend(self.by_text.get(rest[:n], []))
            tid = self.pick(ids, allowed)
            ids.append(tid)
            text += self.tokens[tid] or ""

    def generate_string(self, ids: list[int]) -> str:
        """Opening quote is already in ids. Returns the decoded value.

        The closing quote is NOT added: the caller forces it.
        """
        raw = ""
        for _ in range(MAX_STRING_TOKENS):
            lg = self.logits(ids)
            for tid in np.argsort(-lg):
                tid = int(tid)
                tok = self.tokens[tid] if tid < len(self.tokens) else None
                if not tok:
                    continue
                if STR_BODY.fullmatch(raw + tok):
                    raw += tok
                    ids.append(tid)
                    break
                q = tok.find('"')
                if q != -1 and STR_DONE.fullmatch(raw + tok[:q]):
                    return unescape(raw + tok[:q]).lstrip()
            else:
                break
        return unescape(raw).lstrip()

    def generate_number(self, ids: list[int], integer: bool) -> float:
        prefix = INT_PREFIX if integer else NUM_PREFIX
        full = INT_FULL if integer else NUM_FULL
        text = ""
        while len(text) < MAX_NUMBER_CHARS:
            allowed = [
                i for i in self.number_ids
                if prefix.fullmatch(text + (self.tokens[i] or ""))
            ]
            if full.fullmatch(text):
                allowed += self.end_ids
            tid = self.pick(ids, allowed)
            if tid in self.end_set:
                break
            text += self.tokens[tid] or ""
            ids.append(tid)
        text = text.rstrip(".")
        if text in ("", "-"):
            text = "0"
        return int(text) if integer else float(text)

    def decode_call(self, request: str,
                    schemas: list[dict[str, Any]]) -> dict[str, Any]:
        ids = self.encode(build_prompt(request, schemas))
        ids += self.encode('{"name": "')
        names = list(dict.fromkeys(s["name"] for s in schemas))
        name = self.choose(ids, [n + '"' for n in names])[:-1]
        schema = next(s for s in schemas if s["name"] == name)

        params: dict[str, typing.Any] = {}
        forced = ', "parameters": {'
        for idx, (key, val) in enumerate(schema.get("parameters", {}).items()):
            if idx > 0:
                forced += ", "
            forced += f'"{key}": '
            typ = val.get("type")
            enum = val.get("enum")
            if typ in INT_TYPES or typ in NUM_TYPES:
                ids += self.encode(forced)
                params[key] = self.generate_number(ids, typ in INT_TYPES)
                forced = ""
            elif typ in BOOL_TYPES:
                ids += self.encode(forced)
                params[key] = self.choose(ids, ["true", "false"]) == "true"
                forced = ""
            elif enum and all(isinstance(v, str) for v in enum):
                ids += self.encode(forced + '"')
                params[key] = self.choose(ids, [v + '"' for v in enum])[:-1]
                forced = ""
            else:
                ids += self.encode(forced + '"')
                params[key] = self.generate_string(ids)
                forced = '"'
        return {"prompt": request, "name": name, "parameters": params}


def run_constrained_json_generation(
    data: list[dict[str, Any]],
    model: SLM,
    schemas: list[dict[str, Any]],
) -> list[typing.Optional[dict[str, Any]]]:
    decoder = Decoder(model=model, schemas=schemas)
    results: list[typing.Optional[dict[str, Any]]] = []
    for i, entry in enumerate(data, start=1):
        print(f"[{i}] {entry['prompt']}")
        try:
            call = decoder.decode_call(entry["prompt"], schemas)
        except Exception as exc:
            print(f"  error: {exc}", file=sys.stderr)
            results.append(None)
            continue
        print(f" res = {call['name']} {call['parameters']}")
        results.append(call)
    return results
