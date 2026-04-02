# EcoCode

EcoCode is a small **energy-aware LLM routing** CLI. You type a task at the prompt; it estimates complexity, picks a model tier, shows a short **energy dashboard**, then either answers in chat or (when a file is involved) runs a read–fix–write flow over that file.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` in the project root with the API keys your setup needs (e.g. Gemini, OpenRouter, Cerebras).

**File-based fixes** use an MCP filesystem server via **Node** (`npx` on your `PATH`).

## Run the interactive CLI

From the repo root:

```bash
python ecocode.py
```

You get an `ecocode>` prompt. To **re-print the startup instructions** (examples, full-path note, how to quit), type `help` at that prompt and press Enter. To quit, type `exit`, `quit`, or `q`.

### Full path to the file

To apply a change to a specific file, put an **absolute path** to that file in your prompt (not a relative path). Example:

```text
ecocode> fix the bug in /Users/you/project/src/app.py
```

If EcoCode does not detect a file, it treats the message as a normal chat question.

## Evaluation (SWE-bench Lite)

`evaluate.py` runs EcoCode on a filtered SWE-bench Lite slice and can compare against a baseline that always calls the large model.

For that you’ll need three entries in `.env`, spelled exactly like this: `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, and `CEREBRAS_API_KEY`. They line up with the small / medium / large tiers in ecocode mode plus the baseline. The extra Python packages for eval are already listed in `requirements.txt`.

```bash
python evaluate.py --mode ecocode --n 3
python evaluate.py --mode baseline --n 3
python evaluate.py --compare
```

Run `python evaluate.py --help` for `--n`, `--instance`, and the rest. The script’s module docstring and comments cover metrics and how the slice is built.

## Other entry points

- **Routing only (no LLM call):** `python logic.py "fix the bug in /path/to/file.py"`

Routing and energy estimates are implemented in `logic.py`; the eval harness is `evaluate.py`.
