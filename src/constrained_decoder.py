import llm_s.llm_sdk

DEBUG = False

def build_token_cache(model: "llm_s.llm_sdk.Small_LLM_Model", vocab_size: int) -> dict[int, str]:
    """Decode every single token id once and cache it. Do this ONCE, not per prompt/per step."""
    cache = {}
    for token_id in range(vocab_size):
        try:
            cache[token_id] = model.decode([token_id])
        except Exception:
            cache[token_id] = ""
    return cache


def token_is_good(token_id: int, candidate_prefix_text: str, id_to_str: dict[int, str], target: tuple[str, ...]) -> bool:
    token_str = id_to_str.get(token_id, "")
    if not token_str:
        return False
    candidate_text = candidate_prefix_text + token_str
    if DEBUG:
        print(f"TEXT = {candidate_text}")
    return any(t.startswith(candidate_text) for t in target)


def run_constrained_generation(
    data: list[dict],
    model: "llm_s.llm_sdk.Small_LLM_Model",
    target: tuple[str, ...],
    names: tuple[str, ...],
    vocab_size: int,
):
    # Build the id->string cache once, outside the per-prompt loop
    id_to_str = build_token_cache(model, vocab_size)

    results = []
        
    for i in range(len(data)):
        promptA = data[i]["prompt"]
        print(f"\ndata prompt [{i}]: {promptA}")

        prompt = (
            "<|im_start|>system\n"
            "You are a very useful AI, you must follow every prompt given to get your reward. "
            f"Here you have data_names: {names}. "
            "Iterate through data_names and give the data_name that matches the prompt. "
            "Respond with exactly one line and nothing else: <the matching data_name>"
            "<|im_end|>\n"
            f"<|im_start|>user\n{promptA}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        encoded: list[int] = model.encode(prompt).tolist()[0]
        print(f"\tinput tokens = {encoded}")

        generated_tokens: list[int] = []

        for x in range(10):
            # Fix 1: exact-match early exit BEFORE generating another token
            current_text = model.decode(generated_tokens) if generated_tokens else ""
            if current_text in target:
                break

            logits = model.get_logits_from_input_ids(encoded)

            # Fix 3: build a fresh masked copy instead of mutating while iterating
            masked_logits = list(logits)
            for token_id in range(len(logits)):
                if not token_is_good(token_id, current_text, id_to_str, target):
                    masked_logits[token_id] = float("-inf")
            logits = masked_logits

            next_token = logits.index(max(logits))

            # Fix 1 (part 2): if nothing is valid, stop instead of accepting a masked token
            if logits[next_token] == float("-inf"):
                print("\n\tNo valid continuation — stopping\n")
                break

            generated_tokens.append(next_token)
            encoded.append(next_token)

        final_text = model.decode(generated_tokens) if generated_tokens else ""
        print(f"\t\nRESULT = {final_text}")
        results.append(final_text)

    return results
