import sys
import llm_sdk.llm_sdk


def main() -> None:
    model = llm_sdk.llm_sdk.Small_LLM_Model()
    path = model.get_path_to_tokenizer_file()
    print(path)


if __name__ == "__main__":
    main()
