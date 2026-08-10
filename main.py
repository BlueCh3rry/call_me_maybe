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
        print(f"data prompt [{i}]: {prompt}")
        encoded: list[int] = model.encode(prompt).tolist()[0]
        print(f"MEEEEEEEEee {encoded}")
        lst_n = []
        for x in range(20):
            logits = model.get_logits_from_input_ids(encoded)
            high_nbr = logits.index(max(logits))
            lst_n.append(high_nbr)
        print(f"high nbr lst = {lst_n}\n")
        try:
            with open(model.get_path_to_vocab_file(), "r") as vocab_file:
                vocab: dict[str, int] = json.load(vocab_file)
                keys = list(vocab.keys())
                tok: list[int] = [vocab[keys[val]] for val in lst_n]
                print(f"tok= {tok}")
                decoded = [model.decode(tok)]
                print(f"decoded= {decoded}")
                prompt += decoded
        except FileNotFoundError:
            print("Error: 'data.json' file was not found.")

        print(f"final prompt {prompt}")

    # output()


if __name__ == "__main__":
    main()
