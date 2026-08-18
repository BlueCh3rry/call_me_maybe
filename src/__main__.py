import argparse
import llm_s.llm_sdk
import json
import re


# ANSWER_REGEX = re.compile(r"^Answer:\s*[A-Za-z0-9 ,.!?'\"()-]*$")


def token_is_good(token_id: int, generated_tokens: list[int], model, data_def: dict[int, str]) -> bool:
    candidate_tokens = generated_tokens + [token_id]
    text = model.decode(candidate_tokens)
    print("TEXT = %s", text)
    # Don't allow more than 20 words
    if len(text.split()) > 20:
        return False

    for i in range(len(data_def)):
        if not text.startswith(data_def[i]["name"]):
            return data_def[i]["name"].startswith(text)

    return False
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
    for i in range(len(data_def)):
        names += data_def[i]["name"] + ", "
        if i == len(data_def):
            names += data_def[i]["name"]
    print(names)
    for i in range(len(data)):
        promptA = data[i]["prompt"]
        print(f"\ndata prompt [{i}]: {promptA}")
        prompt = (
            "<|im_start|>system\n"
            "You are a very useful AI, you must follow every prompt given do get your reward."
            f"Your goal is to compare the prompt given to the data_definitions: {names}."
            "Give the name which matchs with the prompt, give only the name matching."
            "If name found print it and that's it you have done your job."
            "Do not do any sentence and no repeat. Do not go over 2 words.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{promptA}\n"
            "<|im_end|>\n"
            # "<|im_start|>assistance\n"
        )

        encoded: list[int] = model.encode(prompt).tolist()[0]
        print(f"\tinput tokens = {encoded}")

        generated_tokens: list[int] = []
        for x in range(60):
            logits = model.get_logits_from_input_ids(encoded)

            # for token_id, logit in enumerate(logits):
            #     if not token_is_good(token_id, generated_tokens, model, data_def):
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
