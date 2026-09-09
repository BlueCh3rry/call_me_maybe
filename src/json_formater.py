import json
import llm_s.llm_sdk
import typing
from src.constrained_decoder import build_token_cache


DEBUG = False

# ---------- JSON prefix validator ----------


class IncompleteJSON(Exception):
    pass


class InvalidJSON(Exception):
    pass


def expect(text, pos, expected):
    end = pos + len(expected)
    actual = text[pos:end]
    if actual == expected:
        return end
    if expected.startswith(actual):
        raise IncompleteJSON()
    raise InvalidJSON(
        f"expected {expected!r}, got {actual!r}"
    )



def match(text, pos, expected):
    if isinstance(expected, str):
        expected = [expected]
    remaining = text[pos:]
    for option in expected:
        if remaining.startswith(option):
            return pos + len(option), option
    for option in expected:
        if option.startswith(remaining):
            raise IncompleteJSON()
    raise InvalidJSON(
        f"expected one of {expected}, got {remaining!r}"
    )


def string(text, pos, allowed=None):
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] != '"':
        raise InvalidJSON(f'expected " at {pos}')
    pos += 1
    out = []
    while pos < len(text):
        c = text[pos]
        pos += 1
        if c == '"':
            value = "".join(out)
            if allowed is not None and value not in allowed:
                raise InvalidJSON(f"{value!r} not allowed")
            return pos, value
        if c == "\\":
            # FIX 1: when `allowed` is a fixed enum of plain identifiers
            # (e.g. function names), a backslash can never legitimately
            # appear. Without this, a stray "\" gets classified as
            # "incomplete" (not invalid) and the model is free to keep
            # bolting on garbage characters one token at a time until it
            # finally accumulates 4 chars for a \uXXXX check and only
            # then fails — by which point several bad tokens are already
            # locked into the generated sequence.
            if allowed is not None:
                raise InvalidJSON(
                    "escape sequence not valid inside enum-constrained string"
                )
            if pos >= len(text):
                raise IncompleteJSON()
            c = text[pos]
            pos += 1
            escapes = {
                '"': '"', "\\": "\\", "/": "/",
                "b": "\b", "f": "\f", "n": "\n",
                "r": "\r", "t": "\t",
            }
            if c in escapes:
                out.append(escapes[c])
            elif c == "u":
                if pos + 4 > len(text):
                    raise IncompleteJSON()
                h = text[pos:pos + 4]
                if not all(x in "0123456789abcdefABCDEF" for x in h):
                    raise InvalidJSON("bad unicode escape")
                out.append(chr(int(h, 16)))
                pos += 4
            else:
                raise InvalidJSON(f"bad escape \\{c}")
        else:
            out.append(c)
        if allowed is not None:
            value = "".join(out)
            if not any(x.startswith(value) for x in allowed):
                raise InvalidJSON(f"{value!r} cannot continue")
    raise IncompleteJSON()


def number(text, pos, max_digits=17):
    # FIX 2: cap how many digits we'll accept. A trailing run of digits
    # at the end of the buffer is always ambiguous (the model could
    # always emit one more), so without a cap it's *always* classified
    # "incomplete" and the digit token is always "allowed" — which is
    # exactly the condition that lets a weak/greedy model loop forever
    # on "0". Once the cap is exceeded we raise InvalidJSON, which masks
    # out further digits and pushes the model toward "," or "}".
    start = pos
    if pos < len(text) and text[pos] == "-":
        pos += 1
    digit_start = pos
    while pos < len(text) and text[pos].isdigit():
        pos += 1
        if pos - digit_start > max_digits:
            raise InvalidJSON("number too long")
    if pos < len(text) and text[pos] == ".":
        pos += 1
        frac_start = pos
        while pos < len(text) and text[pos].isdigit():
            pos += 1
            if pos - frac_start > max_digits:
                raise InvalidJSON("number too long")
    if pos < len(text) and text[pos] in "eE":
        pos += 1
        if pos < len(text) and text[pos] in "+-":
            pos += 1
        while pos < len(text) and text[pos].isdigit():
            pos += 1
    if pos == start:
        raise InvalidJSON("expected number")
    if pos == len(text):
        raise IncompleteJSON()
    return pos


def value(text, pos, schema):
    if pos >= len(text):
        raise IncompleteJSON()
    typ = schema["type"]
    if typ in ("str", "string"):
        pos, _ = string(text, pos)
        return pos
    if typ in ("int", "integer", "float", "number"):
        return number(text, pos)
    if typ in ("bool", "boolean"):
        pos, _ = match(text, pos, ["true", "false"])
        return pos
    raise InvalidJSON(f"unsupported type: {typ}")



def parse_generation(text, schemas, prompt):
    pos = 0
    pos = expect(text, pos, "{")
    # NOTE: "prompt" field intentionally not required here — see FIX 3
    # below, which changes the system prompt to match this parser
    # instead of the other way around. If you'd rather keep the
    # "prompt" field in the output, re-enable these four lines instead
    # of editing the system prompt.
    pos = expect(text, pos, '"prompt"')
    pos = expect(text, pos, ":")
    pos, _ = string(text, pos, [prompt])
    pos = expect(text, pos, ",")
    pos = expect(text, pos, '"name"')
    pos = expect(text, pos, ":")
    names = [s["name"] for s in schemas]
    pos, name = string(text, pos, names)
    schema = next(
        (s for s in schemas if s["name"] == name),
        None,
    )
    if schema is None:
        raise InvalidJSON("unknown function")

    pos = expect(text, pos, ",")
    pos = expect(text, pos, '"parameters"')
    pos = expect(text, pos, ":")
    pos =  expect(text, pos, "{")

    params = schema["parameters"]
    remaining = set(params)

    if pos < len(text) and text[pos] == "}":
        pos += 1
    else:
        while True:
            if not remaining:
                raise InvalidJSON("no params")
            pos, key = string(text, pos, list(remaining))
            remaining.remove(key)
            if pos >= len(text):
                raise IncompleteJSON()
            if text[pos] != ":":
                raise InvalidJSON("expected ':'")
            pos += 1
            # parse taking schema type from functions_def
            pos = value(text, pos, params[key])
            if pos >= len(text):
                raise IncompleteJSON()
            if text[pos] == ",":
                if not remaining:
                    raise InvalidJSON("trailing comma with no remaining params")
                pos += 1
                if pos >= len(text):
                    raise IncompleteJSON()
                if text[pos] == "}":
                    raise InvalidJSON("trailing comma")
                continue
            if text[pos] == "}":
                pos += 1
                break
            raise InvalidJSON("expected ',' or '}'")
    if remaining:
        raise InvalidJSON(f"missing: {remaining}")
    if pos != len(text):
        raise InvalidJSON("trailing characters")
    return pos


def classify_json_candidate(text, schemas, prompt):
    try:
        parse_generation(text, schemas, prompt)
        return "complete"
    except IncompleteJSON:
        return "incomplete"
    except InvalidJSON:
        return "invalid"


def json_token_is_good(token_id: int, prefix: str, id_to_str: dict[int, str], schemas: list[dict], prompt: str) -> bool:
    token = id_to_str.get(token_id)
    if not token:
        return False
    status = classify_json_candidate(prefix + token, schemas, prompt)
    return status != "invalid"


def run_constrained_json_generation(data: list[dict], model: llm_s.llm_sdk.Small_LLM_Model, vocab_size: int, schemas: list[dict]):
    if schemas is None:
        print("NO DATA_REF/SCEHMAS")
    id_to_str = build_token_cache(model, vocab_size)

    # test = '{"name":"fn_add_numbers","parameters":{"a":'
    # print("TEEEEEEEEST =", classify_json_candidate(test, schemas))

    results: list[typing.Any] = []
    for i in range(len(data)):
        promptA: str = data[i]["prompt"]
        print(f"\ndata prompt [{i}]: {promptA}")
        prompt = (
            "<|im_start|>system\n"
            "You are a very useful AI, you must follow every prompt given to get your reward. "
            f"Here are the available functions and their parameters: {schemas}, keep also the prompt given in mind. "
            "Respond with JSON format and nothing else, in the form: "
            # FIX 3: the described output shape now matches what        
            # parse_generation actually accepts (no top-level "prompt"
            # field). Previously the model was told to start with
            # {"prompt": ...} but the grammar hard-required "name" to
            # come first, so any token path toward "prompt" was masked
            # to -inf immediately — actively fighting the instructions
            # the model was just given, right at the start of decoding.
            '{"name": "<matching function name>", "parameters": {<param>: <value>, ...}}'
            "\n/no_think<|im_end|>\n"
            f"<|im_start|>user\n{promptA}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        encoded: list[int] = model.encode(prompt).tolist()[0]
        print(f"\tinput tokens = {encoded}")
        generated_tokens: list[int] = []

        for x in range(60):
            current_text = model.decode(generated_tokens)
            status = classify_json_candidate(current_text, schemas, promptA)
            if status == "complete":
                break
            logits = model.get_logits_from_input_ids(encoded)
            good_count = 0
            masked_logits = list(logits)
            for token_id in range(len(logits)):
                if json_token_is_good(token_id, current_text, id_to_str, schemas, promptA):
                    good_count += 1
                else:
                    masked_logits[token_id] = float("-inf")
            logits = masked_logits
            print(f"  step {x}: {good_count} good tokens out of {len(logits)}")
            next_token = logits.index(max(logits))
            print(f"  step {x}: chose token_id={next_token} str={id_to_str.get(next_token)} logit={logits[next_token]}")
            if logits[next_token] == float("-inf"):
                print("\n\tNo valid continuation — stopping\n")
                break
            generated_tokens.append(next_token)

            print(f"  step {x}: current_text={current_text!r} |status={status}|\n")

            encoded.append(next_token)
        print(f"generated_tokens = {generated_tokens}")
        final_text = model.decode(generated_tokens)
        print(f"\t\nRESULT = {final_text}")
        results.append(final_text)
    return results