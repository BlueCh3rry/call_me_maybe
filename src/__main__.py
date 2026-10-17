# import argparse
# import llm_s.llm_sdk
# import json
# import re


# # ANSWER_REGEX = re.compile(r"^Answer:\s*[A-Za-z0-9 ,.!?'\"()-]*$")


# def main2() -> None:
#     print("Call_me_maybe_MAIN_2")
#     path_tests = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/function_calling_tests.json"
#     path_def = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/functions_definition.json"
#     # path_def = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/functions_definition.json"
#     # path_tests = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/function_calling_tests.json"
#     try:
#         with open(path_tests, "r", encoding="utf-8") as file:
#             data = json.load(file)
#     except FileNotFoundError:
#         print("Error: 'function_calling_tests.json' file was not found.")
#         return
#     try:
#         with open(path_def, "r", encoding="utf-8") as file1:
#             data_def = json.load(file1)
#     except FileNotFoundError:
#         print("Error: 'function_definitions.json' file was not found.")
#         return
#     from src.json_formater import run_constrained_json_generation
#     target = (
#         "fn_add_numbers",
#         "fn_greet",
#         "fn_reverse_string",
#         "fn_get_square_root",
#         "fn_substitute_string_with_regex",
#     )
#     # from src.constrained_decoder import run_constrained_generation
#     # gen = run_constrained_generation(data, llm_s.llm_sdk.Small_LLM_Model(), target, target, 151642)
#     # print("gen = ", gen)

#     # schemas_json = json.dumps(data_def, indent=2)
#     # names2 = tuple(schema["name"] for schema in data_def)
#     jsone = run_constrained_json_generation(data, llm_s.llm_sdk.Small_LLM_Model(), 151642, data_def)
#     print("json = ", jsone)
#     path_output = "C:/Users/Red/home/42Cursus/call_me_maybe/data/output/outputfile.json"

#     results = []
#     for prompt_entry, raw_text in zip(data, jsone):
#         try:
#             parsed_call = json.loads(raw_text)
#         except json.JSONDecodeError as e:
#             print(f"Failed to parse model output for prompt {prompt_entry['prompt']!r}: {e}")
#             continue  # or decide how you want partial failures handled — see note below
#         results.append({
#             "prompt": prompt_entry["prompt"],
#             "name": parsed_call["name"],
#             "parameters": parsed_call["parameters"],
#         })

#     with open(path_output, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)


# if __name__ == "__main__":
#     main2()



import argparse
import json
import sys
from pathlib import Path

import llm_s.llm_sdk
from src.json import run_constrained_json_generation

VALID_TYPES = {"str", "string", "int", "integer", "float", "number", "bool", "boolean"}


def load_json(path: str, what: str):
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


def validate_definitions(raw) -> list[dict]:
    if not isinstance(raw, list):
        sys.exit("Error: function definitions must be a JSON list.")
    good, seen = [], set()  # CHNG: seen = duplicate names
    for idx, fn in enumerate(raw):
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            print(f"Warning: definition #{idx} skipped (needs a string 'name').",
                  file=sys.stderr)
            continue
        if fn["name"] in seen:  # CHNG
            print(f"Warning: duplicate definition '{fn['name']}' skipped.",
                  file=sys.stderr)
            continue
        params = fn.get("parameters", {})
        if not isinstance(params, dict) or not all(
            isinstance(p, dict) and p.get("type") in VALID_TYPES
            for p in params.values()
        ):
            print(f"Warning: definition '{fn['name']}' skipped "
                  "(bad 'parameters').", file=sys.stderr)
            continue
        seen.add(fn["name"])  # CHNG
        good.append({**fn, "parameters": params})
    if not good:
        sys.exit("Error: no usable function definitions.")
    return good


def validate_prompts(raw) -> list[dict]:
    if not isinstance(raw, list):
        sys.exit("Error: input file must be a JSON list of {\"prompt\": ...}.")
    good = []
    for idx, entry in enumerate(raw):
        if (isinstance(entry, dict) and isinstance(entry.get("prompt"), str)
                and entry["prompt"].strip()):
            good.append(entry)
        else:
            print(f"Warning: prompt #{idx} skipped (needs non-empty 'prompt').",
                  file=sys.stderr)
    return good


def parse_args():
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
        model = llm_s.llm_sdk.Small_LLM_Model()
    except Exception as e:
        sys.exit(f"Error: could not load model: {e}")

    # CHNG: no VOCAB_SIZE, returns dicts (or None) instead of raw text
    calls = run_constrained_json_generation(data, model, schemas)

    results = [c for c in calls if c is not None]  # CHNG: no json.loads needed
    if len(results) < len(data):  # CHNG
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
    try:  # CHNG
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)