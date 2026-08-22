import argparse
import llm_s.llm_sdk
import json
import re


# ANSWER_REGEX = re.compile(r"^Answer:\s*[A-Za-z0-9 ,.!?'\"()-]*$")


def token_is_good(token_id: int, generated_tokens: list[int], model: llm_s.llm_sdk.Small_LLM_Model) -> bool:
    candidate_tokens = generated_tokens + [token_id]
    target = ("fn_add_numbers", "fn_greet", "fn_reverse_string", "fn_get_square_root", "fn_substitute_string_with_regex")
    text = model.decode(candidate_tokens)
    print(f"TEXT ={text}")
    return any(t.startswith(text) for t in target)
    # return ANSWER_REGEX.fullmatch(text) is not None


def main() -> None:
    print("Call_me_maybe")
    # argparse
    path_def = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/functions_definition.json"
    path_tests = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/function_calling_tests.json"
    # path = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/function_calling_tests.json"
    try:
        with open(path_tests, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: 'function_calling_tests.json' file was not found.")

    try:
        with open(path_def, "r", encoding="utf-8") as file1:
            data_def = json.load(file1)
    except FileNotFoundError:
        print("Error: 'function_definitions.json' file was not found.")

    model = llm_s.llm_sdk.Small_LLM_Model()
    names = ""
    # parameters = ""
    for i in range(len(data_def)):
        if i == len(data_def) - 1:
            names += data_def[i]["name"]
            # parameters += str(data_def[i]["parameters"])
        else:
            names += data_def[i]["name"] + ", "
            # parameters += str(data_def[i]["parameters"]) + ", "
    print(f"\n{names}\n")
    # print(f"\n{parameters}\n")
    for i in range(len(data)):
        promptA = data[i]["prompt"]
        print(f"\ndata prompt [{i}]: {promptA}")
        prompt = (
            "<|im_start|>start\n"
            "<|im_start|>system\n"
            "You are a very useful AI, you must follow every prompt given to get your reward"
            f"Here you have data_names: {names}."
            # f"Here you have data_parameters: {parameters}."
            "Iterate through data_names and give the data_names which matchs with the prompt."
            # "Iterate through data_parameters same index coresponding as data_names."
            "Respond with exactly one lines and nothing else and stop thinking: <the matching data_name>"
            # "Line 2: <the data_parameter at the same index>"
            # "<|im_end|>\n"
            f"<|im_start|>user\n{promptA}\n/no_think<|im_end|>\n"
            # "<|im_start|>think/assistance\n"
        )

        encoded: list[int] = model.encode(prompt).tolist()[0]
        print(f"\tinput tokens = {encoded}")

        generated_tokens: list[int] = []
        for x in range(40):
            logits = model.get_logits_from_input_ids(encoded)

            # for token_id, logit in enumerate(logits):
            #     if not token_is_good(token_id, generated_tokens, model):
            #         logits[token_id] = float("-inf")

            next_token = logits.index(max(logits))
            # if next_token > 151642:
            #     print("\n\tEOS\n")
            #     break

            generated_tokens.append(next_token)
            encoded.append(next_token)

            # keys = list(vocab.values())
            generated_text = model.decode(generated_tokens)
            print(f"generated = {generated_text} |END")
    # output()


if __name__ == "__main__":
    main()
