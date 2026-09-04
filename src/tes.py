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


def expect_target_prefix(text, pos, Target):
    for word in Target:
        if pos >= len(text):
            raise IncompleteJSON()
        if text[pos] != word:
            raise InvalidJSON(
                f"expected {word}, got {text[pos]}"
            )
        pos += 1
    return pos


def match_literal_prefix(text, pos, options):
    start = pos
    while True:
        consumed = text[start:pos]
        if consumed in options:
            return pos, consumed
        if pos >= len(text):
            raise IncompleteJSON()
        if not any(option.startswith(consumed) for option in options):
            raise InvalidJSON(
                f"{consumed} matches none of {options}"
            )
        pos += 1


def check_valid_word(text: str, pos: int, valid_words: list[str] | None=None):
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] != '"':
        raise InvalidJSON(f'expected \'"\' at {pos}')
    pos += 1
    out: list[str] = []
    while True:
        if pos >= len(text):
            raise IncompleteJSON()
        char = text[pos]
        pos += 1
        if char == '"':
            content = "".join(out)
            if valid_words is not None and content not in valid_words:
                raise InvalidJSON(f"{content} is not an allowed value")
            return pos, content
        if char == "\\":
            if pos >= len(text):
                raise IncompleteJSON()
            esc = text[pos]
            pos += 1
            simple = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if esc in simple:
                out.append(simple[esc])
            elif esc == "u":
                if pos + 4 > len(text):
                    raise IncompleteJSON()
                hex_digits = text[pos:pos + 4]
                if not all(c in "0123456789abcdefABCDEF" for c in hex_digits):
                    raise InvalidJSON("bad unicode escape")
                out.append(chr(int(hex_digits, 16)))
                pos += 4
            else:
                raise InvalidJSON(f"bad escape \\{esc}")
        else:
            out.append(char)
        content = "".join(out)
        if valid_words is not None:
            if not any(word.startswith(content) for word in valid_words):
                raise InvalidJSON(f"{content} cannot lead to allowed value")


def check_skip_digits(text: str, pos: int) -> tuple[int, bool]:
    start = pos
    while pos < len(text) and text[pos].isdigit():
        pos += 1
    return pos, pos > start


def skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos


def parse_number(text, pos):
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] == "-":
        pos += 1
        if pos >= len(text):
            raise IncompleteJSON()
    pos, has_digits = check_skip_digits(text, pos)
    if not has_digits:
        raise InvalidJSON("expected digits")
    if pos < len(text) and text[pos] == ".":
        pos += 1
        pos, has_digits = check_skip_digits(text, pos)
        if not has_digits:
            raise IncompleteJSON()
    if pos < len(text) and text[pos] in "eE":
        pos += 1
        if pos < len(text) and text[pos] in "+-":
            pos += 1
        pos, has_digits = check_skip_digits(text, pos)
        if not has_digits:
            raise IncompleteJSON()
    return pos


def param_value(text, pos, value_type):
    if value_type in ("number", "integer", "int", "float"):
        return parse_number(text, pos)
    if value_type in ("string", "str"):
        return check_valid_word(text, pos)
    if value_type in ("boolean", "bool"):
        return match_literal_prefix(text, pos, ["true", "false"])
    raise InvalidJSON(f"unsupported type {value_type}")


def parse_generation(text, schemas):
    pos = 0
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] != "{":
        raise InvalidJSON("expected {")
    pos += 1
    # pos = skip_ws(text, pos)
    pos = expect_target_prefix(text, pos, '"name"')
    # pos = skip_ws(text, pos)
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] != ":":
        raise InvalidJSON("expected ':'")
    pos += 1
    # pos = skip_ws(text, pos)
    names = tuple(schema["name"] for schema in schemas)
    pos, name = check_valid_word(text, pos, list(names))
    schema = next((schema for schema in schemas if schema["name"] == name), None)
    if schema is None:
        raise InvalidJSON(f"unknown function {name}")
    # pos = skip_ws(text, pos)
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] != ",":
        raise InvalidJSON("expected ','")
    pos += 1
    # pos = skip_ws(text, pos)
    pos = expect_target_prefix(text, pos, '"parameters"')
    # pos = skip_ws(text, pos)
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] != ":":
        raise InvalidJSON("expected ':'")
    pos += 1
    # pos = skip_ws(text, pos)
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] != "{":
        raise InvalidJSON("expected '{'")
    pos += 1
    # pos = skip_ws(text, pos)
    remaining = set(schema["parameters"].keys())
    if pos < len(text) and text[pos] == "}":
        pos += 1
    else:
        while True:
            # pos = skip_ws(text, pos)
            if pos >= len(text):
                raise IncompleteJSON()
            pos, key = check_valid_word(text, pos, list(remaining))
            remaining.remove(key)
            # pos = skip_ws(text, pos)
            if pos >= len(text):
                raise IncompleteJSON()
            if text[pos] == ":":
                raise IncompleteJSON("expected ':'")
            pos += 1
            # pos = skip_ws(text, pos)
            pos = param_value(text, pos, schema["parameters"][key])
            # pos = skip_ws(text, pos)
            if pos >= len(text):
                raise IncompleteJSON()
            if text[pos] == ",":
                pos += 1
                # pos = skip_ws(text, pos)
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
        raise InvalidJSON(f"missing required parameters: {sorted(remaining)}")
    # pos = skip_ws(text, pos)
    if pos >= len(text):
        raise IncompleteJSON()
    if text[pos] != "}":
        raise InvalidJSON("expected final '}'")
    pos += 1
    # pos = skip_ws(text, pos)
    if pos != len(text):
        raise InvalidJSON("trailing characters")
    return pos


def classify_json_candidate(text, schemas):
    try:
        parse_generation(text, schemas)
        return "complete"
    except IncompleteJSON:
        return "incomplete"
    except InvalidJSON as e:
        if DEBUG:
            print(f"INVALID {text}: {e}")
        return "invalid string"


# def json_token_is_good(token_id: int, candidate_prefix_text: str, id_to_str: dict[int, str], names: tuple[str, ...], schemas: list[dict]) -> bool:
#     token_str = id_to_str.get(token_id, "")
#     if not token_str:
#         return False
#     candidate_text = candidate_prefix_text + token_str
#     status = classify_json_candidate(candidate_text, names, schemas)
#     if DEBUG:
#         print(f"TEXT = {candidate_text} -> {status}")
#     return status in ("incomplete", "complete")


def json_token_is_good(token_id: int, candidate_prefix_text: str, id_to_str: dict[int, str], schemas: list[dict]) -> bool:
    token_str = id_to_str.get(token_id, "")
    if not token_str:
        return False
    if (candidate_prefix_text and candidate_prefix_text[-1] in " \t\r\n"):
        return False
    candidate_text = candidate_prefix_text + token_str
    status = classify_json_candidate(candidate_text, schemas)
    if DEBUG:
        print(
            f"token={token_id} "
            f"str={token_str!r} "
            f"candidate={candidate_text!r} "
            f"status={status}"
        )
    return status in ("incomplete", "complete")


def run_constrained_json_generation(data: list[dict], model: llm_s.llm_sdk.Small_LLM_Model, vocab_size: int, schemas: list[dict]):
    if schemas is None:
        print("NO DATA_REF/SCEHMAS")
    from src.constrained_decoder import build_token_cache
    id_to_str = build_token_cache(model, vocab_size)


    test = '{"name":"fn_add_numbers","parameters":{"a":'
    print("TEEEEEEEEST =",classify_json_candidate(test, schemas))

    results: list[dict[str, typing.Any]] = []
    for i in range(len(data)):
        promptA = data[i]["prompt"]
        print(f"\ndata prompt [{i}]: {promptA}")
        prompt = (
            "<|im_start|>system\n"
            "You are a very useful AI, you must follow every prompt given to get your reward. "
            f"Here are the available functions and their parameters: {schemas}. "
            "Respond with JSON format and nothing else, in the form: "
            '{"prompt": "<prompt given>", "name": "<matching function name>", "parameters": {<param>: <value>, ...}}'
            "\n/no_think<|im_end|>\n"
            f"<|im_start|>user\n{promptA}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        encoded: list[int] = model.encode(prompt).tolist()[0]
        print(f"\tinput tokens = {encoded}")
        generated_tokens: list[int] = []

        for x in range(40):
            current_text = model.decode(generated_tokens)
            status = classify_json_candidate(current_text, schemas)
            if status == "complete":
                break
            logits = model.get_logits_from_input_ids(encoded)
            good_count = 0
            masked_logits = list(logits)
            for token_id in range(len(logits)):
                if json_token_is_good(token_id, current_text, id_to_str, schemas):
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
        try:
            parsed = json.loads(final_text)
            path_output = "C:/Users/Red/home/42Cursus/call_me_maybe/data/output/outputfile.json"
            with open(path_output, "w", encoding="utf-8") as f:
                print("OPENINNNNNNNNNNNNNGGGGGGGGGGGG FIIIIIIIIIIIIILEEEEEEEEEEEEEEEEEEE:", path_output)
                json.dump(parsed, f, indent=2)

        except Exception:
            parsed = None
        results.append({"raw": final_text, "parsed": parsed})
    return results
