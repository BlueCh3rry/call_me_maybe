import argparse
import llm_sdk
import json


def output() -> None:
    path = "/home/mmakhmae/sgoinfre/call_me_maybe/data/output/function_calling_results.json"
    # with open(path, "r") as file:
        # outputfile = json.

def main() -> None:
    print("Call_me_maybe")
    # argparse
    path = "/home/mmakhmae/sgoinfre/call_me_maybe/data/input/function_calling_tests.json"
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: 'data.json' file was not found.")

    model = llm_sdk.Small_LLM_Model()
    for i in range(len(data)):
        prompt = data[i]['prompt']
        print(f"data prompt [{i}]: {prompt}\n")
        encoded: list[int] = model.encode(prompt).tolist()[0]
        # path_voc = model.get_path_to_vocab_file()
        # try:
        #     with open(path_voc, "r") as vocab_file:
        #         vocab = json.load(vocab_file)
        # except FileNotFoundError:
        #     print("Error: 'data.json' file was not found.")
        # for _ in range(120):
        print(f"\nMEEEEEEEEee {encoded}\n")
        logits = model.get_logits_from_input_ids(encoded)
        high_nbr = max(logits)
        print(f"high nbr = {high_nbr}\n")

    # output()


if __name__ == "__main__":
    main()
