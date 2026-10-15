import argparse
import llm_s.llm_sdk
import json
import re


# ANSWER_REGEX = re.compile(r"^Answer:\s*[A-Za-z0-9 ,.!?'\"()-]*$")


def main2() -> None:
    print("Call_me_maybe_MAIN_2")
    path_tests = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/function_calling_tests.json"
    path_def = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/functions_definition.json"
    # path_def = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/functions_definition.json"
    # path_tests = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/function_calling_tests.json"
    try:
        with open(path_tests, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: 'function_calling_tests.json' file was not found.")
        return
    try:
        with open(path_def, "r", encoding="utf-8") as file1:
            data_def = json.load(file1)
    except FileNotFoundError:
        print("Error: 'function_definitions.json' file was not found.")
        return
    from src.json_formater import run_constrained_json_generation
    target = (
        "fn_add_numbers",
        "fn_greet",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
    )
    # from src.constrained_decoder import run_constrained_generation
    # gen = run_constrained_generation(data, llm_s.llm_sdk.Small_LLM_Model(), target, target, 151642)
    # print("gen = ", gen)

    # schemas_json = json.dumps(data_def, indent=2)
    # names2 = tuple(schema["name"] for schema in data_def)
    jsone = run_constrained_json_generation(data, llm_s.llm_sdk.Small_LLM_Model(), 151642, data_def)
    print("json = ", jsone)
    path_output = "C:/Users/Red/home/42Cursus/call_me_maybe/data/output/outputfile.json"

    results = []
    for prompt_entry, raw_text in zip(data, jsone):
        try:
            parsed_call = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse model output for prompt {prompt_entry['prompt']!r}: {e}")
            continue  # or decide how you want partial failures handled — see note below
        results.append({
            "prompt": prompt_entry["prompt"],
            "name": parsed_call["name"],
            "parameters": parsed_call["parameters"],
        })

    with open(path_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main2()
