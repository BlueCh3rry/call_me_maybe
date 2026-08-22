class ConstrainedDecoder:
    def __init__(self, model, vocab, schema):
        self.model = model
        self.vocab = vocab
        self.schema = schema

    def generate(self, prompt, max_tokens=256):
        input_ids = self.model.encode(prompt).tolist()
        generated = ""
        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            valid_tokens = self.get_valid_tokens(generated)
            constrained_logits = self.apply_constraints(
                logits,
                valid_tokens,
            )
            token_id = self.select_token(constrained_logits)
            if self.is_eos(token_id):
                break
            input_ids.append(token_id)
            token = self.vocab[token_id]
            generated += token
            if self.is_complete(generated):
                break
        return generated


class FunctionSchema:
    def __init__(self, definition):
        self.name = definition["name"]
        self.description = definition.get("description", "")
        self.parameters = definition.get("parameters", {})
        self.returns = definition.get("returns")


class FunctionRegistry:
    def __init__(self, definitions):
        self.functions = {
            item["name"]: FunctionSchema(item)
            for item in definitions
        }

    def names(self):
        return list(self.functions)

    def get(self, name):
        return self.functions[name]
