*This project has been created as part of the 42 curriculum by mmakhmae.*

## Description

Small language models are unreliable at producing valid JSON when asked
nicely. This project makes model produce a valid function call every time, by controlling the generation token by token instead of trusting the model to follow the format.

The program does not execute anything. It translates the request into a
structured call in json format

## Instructions

### Requirements

- Python 3.10+
- `uv`
- The provided `llm_sdk` package
- `numpy`

### Installation

```bash
uv sync
```

### Execution

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

#### Option

`--functions_definition`\
`--input`\
`--output`

paths by default:

`data/input/functions_definition.json`\
`data/input/function_calling_tests.json`\
`data/output/function_calls.json`

The output folder is created if it does not exist.



## Algorithm explanation

The program uses constrained decoding. At each step the model produces
logits (a score for every token in its vocabulary). Instead of sampling
freely, we mask every token that would break the expected JSON structure by
setting its score to `-inf`, then take the highest remaining token (argmax).
The output can therefore never be malformed.

The generation of one call works in these steps:

1. Prompt: a chat-style prompt lists the available functions and the
   rules, followed by the user request. The assistant turn is then started
   with the forced prefix `{"name": "`.
2. Function name: `choose()` lets the model pick among the valid
   function names. Only tokens that continue one of the remaining names are
   allowed. When a single candidate is left, the rest is auto-completed
   without asking the model.
3. Parameters: the keys, quotes, commas and braces are taken from the
   function's schema and forced into the context. The model only produces the
   values, one by one, according to each type:
   - number / integer: only tokens made of `0-9`, `.` and `-` that keep
     the text a valid numeric prefix are allowed (regex). The tokens that
     start with `,` or `}` become allowed only once the number is complete,
     which is how the model ends it.
   - boolean: `choose()` between `true` and `false`.
   - string enum: `choose()` between the enum values.
   - string: tokens are tried from the most to the least likely, and the
     first one that keeps the text a valid JSON string body (regex) is
     accepted. A token containing the closing quote ends the string. The
     value is then unescaped (`\n`, `\uXXXX`, ...).
4. Result: the program assembles `{"prompt", "name", "parameters"}` and
   writes all results to the output file.

### Design decisions

- Regex prefix checks instead of a JSON parser: validating each candidate
  token by running a JSON parse over the whole vocabulary was the cause of
  the first slow version. Precompiled regexes are much cheaper.
- Structure is forced, not generated: everything that is already known
  (keys, punctuation, the function name once only one option remains) is
  encoded directly and costs no model call.
- Precomputed token lists: the number tokens and the end tokens (`,` and
  `}`) are found once at startup, so number generation only scans a small
  subset of the vocabulary.
- Safety limits: strings are capped at 60 tokens and numbers at 20
  characters, so generation always terminates.
- Robust input handling: missing, unreadable or invalid files give a clear
  error message. Invalid or duplicate function definitions and invalid prompts
  are skipped with a warning. If one prompt fails, the others are still
  processed.

### Performance analysis

- Accuracy: structurally, the output is always valid JSON that matches the
  schema, because invalid tokens are impossible. The correctness of the
  content (right function, right values) depends on the model and the
  prompt.
- Speed: the cost is dominated by one forward pass of the model per
  generated token. Forced text is added without sampling, number generation
  only considers number tokens, and string generation usually accepts one of
  the first few tokens of the sorted logits, so the full vocabulary is rarely
  scanned.
- Reliability: all error paths (bad files, bad schemas, failed prompts)
  are handled without crashing the whole run.

### Challenges faced

- Slow generation: validating the whole vocabulary with a JSON parse at
  each step was far too slow. Replacing it with precompiled regexes and
  precomputed token lists fixed it.
- Strings of unknown length: deciding when a string ends. The solution is
  to accept a token that contains the closing quote only if the text before
  it is a valid complete string.
- Escapes: a token can end in the middle of an escape sequence (a lone
  `\`), which the string regex explicitly allows.
- Unwanted spaces: the model tends to add a leading space at the start of
  a string even though the prompt asks it not to.

### Testing strategy

- Run the program on the provided test file and check the output by hand.
- Check that the output file is valid JSON and every call matches its schema
  (right function name, right parameter names and types).
- Edge cases: missing file, directory instead of file, invalid JSON, empty
  prompt, duplicate function names, unknown parameter types, and an output
  folder that does not exist.
- Prompts with special characters (quotes, backslashes, accents) to test
  string escaping, and negative or decimal numbers to test number generation.

### Example usage

```bash
uv run python -m src
```

Console output while running:

```
[3] Greet shrek
 res = fn_greet {'name': 'shrek'}
```

Output file:

```json
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  }
]
```

## Resources

- Python `argparse`, `re`, `uv` documentations
- geeksforgeeks
- AI assistant was used during development, for: \
Identifying that scanning the full vocabulary with a JSON parse at every step was the cause of the slow run. \
Helping restructure the decoder. \
Helping write and explain the string and number regexes.
