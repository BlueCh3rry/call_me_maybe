import argparse
import llm.llm_sdk
import json
import re



ANSWER_REGEX = re.compile(r"^Answer:\s*[A-Za-z0-9 ,.!?'\"()-]*$")


def token_is_good(token_id: int, generated_tokens: list[int], model) -> bool:
    candidate_tokens = generated_tokens + [token_id]
    text = model.decode(candidate_tokens)

    # Don't allow more than 20 words
    if len(text.split()) > 20:
        return False

    target = ""

    if not text.startswith(target):
        return target.startswith(text)

    return ANSWER_REGEX.fullmatch(text) is not None


def main() -> None:
    print("Call_me_maybe")
    # argparse
    # path = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/function_calling_tests.json"
    path = "C:/Users/Red/home/42Cursus/call_me_maybe/data/input/function_calling_tests.json"
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: 'data.json' file was not found.")

    model = llm.llm_sdk.Small_LLM_Model()
    for i in range(len(data)):
        promptA = data[i]["prompt"]
        print(f"\ndata prompt [{i}]: {promptA}")
        prompt = (
            "<|im_start|>system\n"
            "You are a very useful artificial intelligent"
            "Do not repeat multiple times."
            "Read it carefully."
            "Give a description depending on the prompt given and name it DESC: [description].\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{promptA}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        encoded: list[int] = model.encode(prompt).tolist()[0]
        print(f"\tinput tokens = {encoded}")

        generated_tokens: list[int] = []
        for x in range(20):
            logits = model.get_logits_from_input_ids(encoded)

            for token_id, logit in enumerate(logits):
                if not token_is_good(token_id, generated_tokens, model):
                    logits[token_id] = float("-inf")

            next_token = logits.index(max(logits))
            if next_token > 151642:
                print("\n\tEOS\n")
                break

            generated_tokens.append(next_token) 
            encoded.append(next_token)

            # keys = list(vocab.values())
            generated_text = model.decode(generated_tokens)
            print(f"generated = {generated_text} |END")
    # output()


if __name__ == "__main__":
    main()
