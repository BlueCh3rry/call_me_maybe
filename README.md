*This project has been created as part of the 42 curriculum by mmakhmae.*

## Description

Small language models are unreliable at producing valid JSON when asked
nicely. This project makes a 0.6B-parameter model produce a **valid function
call every time**, by controlling the generation token by token instead of
trusting the model to follow the format.

Given a natural-language request and a list of available functions, the
program does **not** execute anything. It translates the request into a
structured call:

```
"What is the sum of 2 and 3?"
        |
        v
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)
- The provided `llm_sdk` package (wrapper around `Qwen/Qwen3-0.6B`)
- `numpy`

### Installation

```bash
uv sync
```

The model weights are downloaded by the SDK on first use.

### Execution

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |

The output folder is created if it does not exist.

### Input formats

`functions_definition.json`:

```json
[
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": { "name": { "type": "string" } },
    "returns": { "type": "string" }
  }
]
```

Supported parameter types: `string`/`str`, `number`/`float`,
`integer`/`int`, `boolean`/`bool`. A string parameter may also have an
`enum` list of strings.

`function_calling_tests.json`:

```json
[ { "prompt": "Greet shrek" } ]
```

## Example usage

Default paths:

```bash
uv run python -m src
```

Custom files:

```bash
uv run python -m src \
  --functions_definition def.json \
  --input prompts.json \
  --output out.json
```

Console output while running:

```
[3] Greet shrek
 res = fn_greet {'name': 'shrek'}
```

Output file (`out.json`):

```json
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "What is the square root of 16?",
    "name": "fn_get_square_root",
    "parameters": { "a": 16.0 }
  },
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "(\\d+)",
      "replacement": "NUMBERS"
    }
  }
]
```

(`"\\d+"` is JSON escaping; the real regex is `\d+`.)

## Algorithm explanation

### Core principle

At each step the model returns one logit per vocabulary token (about 150k).
Constrained decoding does:

```
logits  ->  keep only ALLOWED token ids  ->  everything else = -inf  ->  argmax
```

A token that is not allowed can never be chosen, so the model cannot leave
the valid format.

### Forced parts vs free parts

The program splits the output in two kinds of text.

| Forced (no model call) | Free (model, masked) |
|---|---|
| `{"name": "` | the function name |
| `", "parameters": {` | each argument value |
| `"key": ` before each argument | |
| the closing quote of a string | |

Forced text is tokenized and appended to the context. The model sees it as
if it had written it, so its next choice is conditioned on a correct
structure.

Context growth for a two-number function:

```
<prompt> {"name": "  [NAME]  ", "parameters": {"a":  [NUMBER]  , "b":  [NUMBER]
         ^forced      ^model   ^forced                ^model      ^forced  ^model
```

### Function name: `choose()`

- The options are the function names, each with a closing `"` appended so
  that `fn_add` can never be taken for the start of `fn_add_numbers`.
- At each step, the allowed tokens are those whose text is a prefix of what
  remains of at least one option (looked up in a dict `text -> ids`).
- **Auto-complete:** as soon as only one option still matches, the rest of it
  is appended with no model call.
- The same routine handles booleans (`true`/`false`) and string enums.

### Numbers: `generate_number()`

- Only tokens made of `0-9 . -` are candidates (precomputed once).
- A candidate is allowed if `text + token` still matches a *prefix* regex
  (`-?(?:\d+(?:\.\d*)?)?` for floats, `-?\d*` for integers).
- When the text is already a *complete* number (`-?\d+(?:\.\d+)?`), tokens
  starting with `,` or `}` become allowed. If the model picks one, it means
  "the number is finished". That token is dropped, and the next forced
  chunk writes the real separator.
- Limit: 20 characters.

### Strings: `generate_string()`

- The opening quote is forced. Tokens are tried from the best logit to the
  worst, and the first valid one wins (usually 1-3 checks, not 150k).
- A token continues the string if `raw + token` matches
  `(?:[^"\\\x00-\x1f]|\\.)*\\?` (no raw quote, no control character,
  backslash escapes allowed).
- A token containing a `"` ends the string: the text before the quote is
  kept, the token is dropped, and the closing quote is forced by the caller.
  This also handles tokens where the quote is glued to other characters.
- Only real JSON escapes (`\" \\ \/ \n \r \t \uXXXX`) are decoded. `\d`,
  `\w`, `\b` stay literal, so regex arguments are not damaged.
- Limit: 60 tokens.

### Reading the vocabulary

Qwen uses byte-level BPE: bytes are stored as stand-in characters (space is
`Ġ`, newline is `Ċ`). The program rebuilds the GPT-2 byte table and decodes
every token to its real UTF-8 text. Tokens that are only half of a
multi-byte character decode to `None` and are never used.

## Design decisions

| Decision | Reason |
|---|---|
| Write the JSON skeleton ourselves | Saves forward passes (the slowest operation) and removes any chance of a structural error. |
| Mask with numpy (`np.full` + index assignment) | One vectorized operation instead of a Python loop over 150k ids. |
| Precompute `by_text`, `number_ids`, `end_ids` once | Each step filters a short list instead of the full vocabulary. |
| Auto-complete when one option is left | The rest of a function name costs zero model calls. |
| End values by token type, not by parsing the whole text | No re-parsing of the JSON on every candidate token. |
| Build the result as a dict | No `json.loads` on model text, so no invalid-JSON failures. |
| Allow backslashes in strings | The regex function needs `\d+`, `\s`, etc. |
| Hard caps per value (60 tokens, 20 chars) | A runaway value cannot loop forever. |
| One `try/except` per prompt | One failing prompt does not stop the others. |
| Generic prompt rules, no example-specific text | The input files may change, so nothing is hardcoded to the examples. |
| Standard-library `re` only | No extra dependency for the constraints. |

The program deliberately does **not** use the model's own sampling. Decoding
is greedy (argmax) over the allowed tokens, so results are deterministic.

## Performance analysis

### Accuracy

On the 11 provided prompts (5 functions: add, greet, reverse, square root,
regex substitution):

- Function selection: **11/11** correct.
- Arguments: **10/11** fully correct on the first run. The single error was a
  replacement value with a leading space (`" *"` instead of `"*"`). A prompt
  rule ("do not add leading or trailing spaces unless asked") was added to
  address it.
- Output structure: valid in every case, by construction.

This is a small, easy test set, so the figures show that the approach works,
not that the accuracy is guaranteed on harder prompts.

### Speed

The first version asked the model to generate every token and checked all
~150k vocabulary tokens with a full JSON parse at each step. That is what
made a run take about 10 minutes. The current version:

- never scans the whole vocabulary,
- spends no forward pass on forced text or on the end of a function name,
- uses numpy for masking.

The remaining cost is the forward passes (the SDK exposes no KV cache, so
each pass processes the whole context again). Roughly 10-20 passes per
prompt are needed instead of 40-60.

Measured total time for the 11 prompts: **`<fill in: X s>`**
(measure with `time.perf_counter()` around `run_constrained_json_generation`).

### Reliability

- Missing, unreadable or invalid input files give a clear error message and a
  non-zero exit code.
- Malformed definitions or prompts are skipped with a warning; if nothing
  usable is left the program stops with an error.
- A failing prompt is reported and skipped, the others still run.
- Per-value caps guarantee termination.
- Ctrl+C exits cleanly (code 130).

## Challenges faced

1. **Run time of about 10 minutes.** The cause was testing every vocabulary
   token with a full JSON parse at every step. Solved by forcing the
   structure, precomputing token tables, masking with numpy and using
   best-logit-first search for strings.
2. **The model copied the prompt into the output token by token.** This wasted
   passes and broke on quotes inside the prompt. Solved by not generating the
   `prompt` field at all and adding it from the input.
3. **Backslashes in regex arguments.** Many constrained decoders forbid `\`
   in strings, which makes `\d+` impossible. Solved by allowing escape pairs
   in the string regex and decoding only real JSON escapes.
4. **Ending a value cleanly.** Tokens such as `."` or `,` mix the end of a
   value with structure characters. Solved by cutting a string at the first
   quote, and by allowing `,`/`}` tokens as the "end" signal for numbers,
   then dropping that token and forcing the real separator.
5. **Byte-level vocabulary.** Tokens contain stand-in characters (`Ġ`, `Ċ`).
   Solved by rebuilding the GPT-2 byte table and decoding each token to UTF-8.
6. **A stray leading space in a string value.** The model preferred the token
   `" *"` over `"*"`. Addressed with a generic prompt rule.
7. **Prefix ambiguity between function names.** `fn_add` versus
   `fn_add_numbers`. Solved by adding the closing quote to every option.

## Testing strategy

- **Functional run** on the provided set of 11 prompts, covering all five
  functions. Output compared by hand with the expected call for each prompt
  (name, each argument, types).
- **Input robustness.** Check these cases and confirm a clean message and exit
  code instead of a traceback:
  - missing file, directory instead of file, unreadable file
  - invalid JSON, empty file, wrong top-level type
  - definition without a `name`, duplicate names, unknown parameter type
  - prompt missing the `prompt` key, empty prompt
- **Change of inputs.** Run with different definition and prompt files to
  confirm nothing depends on the provided examples.
- **Debugging method.** When a result is wrong, check the function name first
  (it is the only free choice before the values), then the value.
- **Timing.** Measure the full run and compare against the 5-minute target.

## Limitations

- Array and object parameters are not supported (rejected during validation).
- Characters split across several byte tokens (some accents or emoji) cannot
  be produced exactly.
- Numbers are written as floats for the `number` type (`2.0`).
- A string value cannot start with whitespace by design of the prompt rule,
  only discouraged, not enforced by the decoder.

## Resources

### References

- Qwen3 model card and documentation: https://huggingface.co/Qwen/Qwen3-0.6B
- Willard & Louf, *Efficient Guided Generation for Large Language Models*
  (2023), https://arxiv.org/abs/2307.09702
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword
  Units* (2016), https://arxiv.org/abs/1508.07909 (BPE)
- Radford et al., *Language Models are Unsupervised Multitask Learners*
  (GPT-2, byte-level BPE), 2019
- Hugging Face, tokenizer summary:
  https://huggingface.co/docs/transformers/tokenizer_summary
- Python `re` module: https://docs.python.org/3/library/re.html
- Python `argparse`: https://docs.python.org/3/library/argparse.html
- NumPy documentation: https://numpy.org/doc/
- JSON specification, RFC 8259: https://www.rfc-editor.org/rfc/rfc8259
- uv documentation: https://docs.astral.sh/uv/

### Use of AI

An AI assistant (Claude) was used during development, for:

- **Performance diagnosis:** identifying that scanning the full vocabulary
  with a JSON parse at every step was the cause of the slow run.
- **Code rewrite:** helping restructure the decoder (forced skeleton,
  precomputed token tables, numpy masking, auto-complete) after comparing it
  with other decoder designs.
- **Regex design:** helping write and explain the string and number regexes.
- **Error handling:** suggesting the input validation and exception structure
  in `__main__.py`.
- **Documentation:** drafting the explanation of the code and this README.

All generated code was read, run on the test inputs and adjusted by the
author, who is able to explain every part of it. Prompt wording and the final
integration choices were made by the author.