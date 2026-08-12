# import argparse
# import llm_sdk
# import json


# def output() -> None:
#     path = "/home/mmakhmae/sgoinfre/call_me_maybe/data/output/function_calling_results.json"
#     # with open(path, "r") as file:
#         # outputfile = json.


# def main() -> None:
#     print("Call_me_maybe")
#     # argparse
#     path = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/function_calling_tests.json"
#     try:
#         with open(path, "r") as file:
#             data = json.load(file)
#     except FileNotFoundError:
#         print("Error: 'data.json' file was not found.")

#     model = llm_sdk.Small_LLM_Model()
#     for i in range(len(data)):
#         prompt = data[i]['prompt']
#         print(f"data prompt [{i}]: {prompt}")
#         encoded: list[int] = model.encode(prompt).tolist()[0]
#         print(f"MEEEEEEEEee {encoded}")
#         lst_n = []
#         for x in range(20):
#             logits = model.get_logits_from_input_ids(encoded)
#             high_nbr = logits.index(max(logits))
#             lst_n.append(high_nbr)
#         print(f"high nbr lst = {lst_n}\n")
#         try:
#             with open(model.get_path_to_vocab_file(), "r") as vocab_file:
#                 vocab: dict[str, int] = json.load(vocab_file)
#                 keys = list(vocab.keys())
#                 tok: list[int] = [vocab[keys[val]] for val in lst_n]
#                 print(f"tok= {tok}")
#                 decoded = [model.decode(tok)]
#                 print(f"decoded= {decoded}")
#                 prompt += decoded
#         except FileNotFoundError:
#             print("Error: 'data.json' file was not found.")

#         print(f"final prompt {prompt}")

#     # output()


# def main() -> None:
#     print("Call_me_maybe")
#     # argparse
#     try:
#         with open(path, "r") as file:
#             data = json.load(file)
#     except FileNotFoundError:
#         print("Error: 'data.json' file was not found.")
#         raise

#     model = llm_sdk.Small_LLM_Model()

#     for i in range(len(data)):
#         prompt = data[i]["prompt"]
#         print(f"data prompt [{i}]: {prompt}")

#         # Start with the tokens for the prompt
#         encoded: list[int] = model.encode(prompt).tolist()[0]
#         print(f"input tokens = {encoded}")

#         generated_tokens = []

#         # Generate 20 new tokens
#         for x in range(20):
#             logits = model.get_logits_from_input_ids(encoded)

#             next_token = logits.index(max(logits))
#             generated_tokens.append(next_token)
#             encoded.append(next_token)

#             print(f"step {x}: token = {next_token}")

#         print(f"generated token ids = {generated_tokens}")

#         try:
#             with open(model.get_path_to_vocab_file(), "r") as vocab_file:
#                 vocab: dict[str, int] = json.load(vocab_file)

#             keys = list(vocab.keys())
#             generated_text = model.decode(generated_tokens)

#             print(f"generated = {generated_text}")

#             final_prompt = prompt + generated_text
#             print(f"final prompt = {final_prompt}")

#         except FileNotFoundError:
#             print("Error: vocab file was not found.")

#         print(f"final prompt {prompt}")

#     # output()


def main() -> None:
    print("Call_me_maybe")

    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: 'data.json' file was not found.")
        raise

    model = llm_sdk.Small_LLM_Model()

    # Load vocabulary BEFORE generation
    with open(model.get_path_to_vocab_file(), "r") as vocab_file:
        vocab: dict[str, int] = json.load(vocab_file)

    # Depending on your vocab format, this may need reversing.
    id_to_token = {token_id: token for token, token_id in vocab.items()}

    # Example constraint
    regex = r"[A-Za-z ]+"

    for i in range(len(data)):
        prompt = data[i]["prompt"]

        print(f"data prompt [{i}]: {prompt}")

        encoded: list[int] = model.encode(prompt).tolist()[0]

        generated_tokens = []

        for x in range(20):
            logits = model.get_logits_from_input_ids(encoded)

            # Text generated so far
            generated_text = model.decode(generated_tokens)

            # Find tokens that don't violate the regex constraint
            allowed_tokens = []

            for token_id, token_text in id_to_token.items():
                candidate = generated_text + token_text

                if regex_allows_prefix(regex, candidate):
                    allowed_tokens.append(token_id)

            if not allowed_tokens:
                print("No valid tokens remain!")
                break

            # Mask invalid tokens
            best_token = max(
                allowed_tokens,
                key=lambda token_id: logits[token_id],
            )

            generated_tokens.append(best_token)
            encoded.append(best_token)

            print(f"step {x}: token = {best_token}")

        generated_text = model.decode(generated_tokens)

        print(f"generated = {generated_text}")
    # output()


if __name__ == "__main__":
    main()
