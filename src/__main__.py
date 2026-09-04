import argparse
import llm_s.llm_sdk
import json
import re


# ANSWER_REGEX = re.compile(r"^Answer:\s*[A-Za-z0-9 ,.!?'\"()-]*$")


# def token_is_good(token_id: int, generated_tokens: list[int], model: llm_s.llm_sdk.Small_LLM_Model, target: list[str]) -> bool:
#     candidate_tokens = generated_tokens + [token_id]
#     print("lst = ", target)
#     # target = ("fn_add_numbers", "fn_greet", "fn_reverse_string", "fn_get_square_root", "fn_substitute_string_with_regex")
#     text = model.decode(candidate_tokens)
#     print(f"TEXT = {text}")
#     return bool(text) and any(t.startswith(text) for t in target)

# def main() -> None:
#     print("Call_me_maybe")
    # argparse
    # path_tests = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/function_calling_tests.json"
    # path_def = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/functions_definition.json"
    # try:
    #     with open(path_tests, "r", encoding="utf-8") as file:
    #         data = json.load(file)
    # except FileNotFoundError:
    #     print("Error: 'function_calling_tests.json' file was not found.")

    # try:
    #     with open(path_def, "r", encoding="utf-8") as file1:
    #         data_def = json.load(file1)
    # except FileNotFoundError:
    #     print("Error: 'function_definitions.json' file was not found.")

    # model = llm_s.llm_sdk.Small_LLM_Model()
    # names = ""
    # lst: list[str] = []
    # # parameters = ""
    # for i in range(len(data_def)):
    #     lst.append(data_def[i]["name"])
    #     if i == len(data_def) - 1:
    #         names += data_def[i]["name"]
    #         # parameters += str(data_def[i]["parameters"])
    #     else:
    #         names += data_def[i]["name"] + ", "
    #         # parameters += str(data_def[i]["parameters"]) + ", "
    # print(f"\n{names}\n")
    # # print(f"\n{parameters}\n")
    # for i in range(len(data)):
    #     promptA = data[i]["prompt"]
    #     print(f"\ndata prompt [{i}]: {promptA}")
    #     prompt = (
    #         "<|im_start|>system\n"
    #         "You are a very useful AI, you must follow every prompt given to get your reward"
    #         f"Here you have data_names: {names}."
    #         # f"Here you have data_parameters: {parameters}."
    #         "Iterate through data_names and give the data_names which matchs with the prompt."
    #         # "Iterate through data_parameters same index coresponding as data_names."
    #         "Respond with exactly one lines and nothing else and stop thinking: <the matching data_name>"
    #         # "Line 2: <the data_parameter at the same index>"
    #         # "<|im_end|>\n"
    #         f"<|im_start|>user\n{promptA}\n/no_think<|im_end|>\n"
    #         # "<|im_start|>think/assistance\n"
    #     )

    #     encoded: list[int] = model.encode(prompt).tolist()[0]
    #     print(f"\tinput tokens = {encoded}")

    #     generated_tokens: list[int] = []
    #     for x in range(10):
    #         current_text = model.decode(generated_tokens) if generated_tokens else ""
    #         if current_text in lst:
    #             break

    #         logits = model.get_logits_from_input_ids(encoded)
    #         for token_id, logit in enumerate(logits):
    #             if not token_is_good(token_id, generated_tokens, model, lst):
    #                 logits[token_id] = float("-inf")

    #         next_token = logits.index(max(logits))
    #         if logits[next_token] == float("-inf"):
    #             # no valid continuation exists — stop instead of picking garbage
    #             break

    #         generated_tokens.append(next_token)
    #         encoded.append(next_token)

    #         # keys = list(vocab.values())
    #         generated_text = model.decode(generated_tokens)
    #         print(f"generated = {generated_text} |END")
    # # output()



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

#     try:
#         with open(path_def, "r", encoding="utf-8") as file1:
#             data_def = json.load(file1)
#     except FileNotFoundError:
#         print("Error: 'function_definitions.json' file was not found.")
#     from src.constrained_decoder import run_constrained_generation
#     from src.tes import run_constrained_json_generation
#     gen = run_constrained_generation(data ,llm_s.llm_sdk.Small_LLM_Model(), target, names, 151642)
#     print("gen = ", gen)
#     jsone = run_constrained_json_generation(data ,llm_s.llm_sdk.Small_LLM_Model(), names2, 151642, schemas=None)
#     print("json = ", jsone)


def main2() -> None:
    print("Call_me_maybe_MAIN_2")
    # path_tests = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/function_calling_tests.json"
    # path_def = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/functions_definition.json"
    path_def = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/functions_definition.json"
    path_tests = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/function_calling_tests.json"
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
    from src.tes import run_constrained_json_generation
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
    # path_output = "C:/Users/Red/home/42Cursus/call_me_maybe/data/output/outputfile.json"
    path_output = "/home/mmakhmae/sgoinfre/call_me_maybe/data/output/outputfile.json"
    with open(path_output, "w", encoding="utf-8") as f:
        print("OPENINNNNNNNNNNNNNGGGGGGGGGGGG FIIIIIIIIIIIIILEEEEEEEEEEEEEEEEEEE:", path_output)
        json.dump(jsone, f, indent=2)


if __name__ == "__main__":
    main2()
