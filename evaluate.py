#!/usr/bin/env python3
"""evaluate.py — Benchmark EcoCode routing vs always-large baseline on SWE-bench Lite.

Evaluation metrics (no pytest required):
  - Hunk BLEU:   sacrebleu BLEU on the lines the gold patch touches (0–100)
  - Patch F1:    precision/recall F1 on the set of added lines vs gold patch (0–1)
  - Exact Match: F1 == 1.0 (every gold-added line reproduced verbatim)

Usage:
    python evaluate.py --mode ecocode --n 10
    python evaluate.py --mode baseline --n 36
    python evaluate.py --mode ecocode --instance pallets__flask-4992
    python evaluate.py --compare
    python evaluate.py --test
"""

import argparse
import asyncio
import difflib
import json
import os
import re
from pathlib import Path

import litellm
import sacrebleu
from dotenv import load_dotenv
from unidiff import PatchSet

from logic import (
    MODEL_ENERGY_COST,
    LARGE_MODEL,
    repo_checker,
    load_full_file,
    route_input,
    InputData,
)

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EcoCode mode: each tier routes to a different model
TIER_TO_MODEL = {
    "small":  "gemini/gemini-2.5-flash",
    "medium": "openrouter/meta-llama/llama-3.3-70b-instruct",
    "large":  "cerebras/qwen-3-235b-a22b-instruct-2507",
}

# Baseline mode: always uses the largest model
BASELINE_MODEL = "cerebras/qwen-3-235b-a22b-instruct-2507"

INTER_CALL_SLEEP      = 3
# Groq often returns Retry-After ~60s on 429; spacing must exceed that or every medium-tier
# call fails after retries. Lower this only if your Groq plan has higher RPM/TPM.
INTER_CALL_SLEEP_GROQ = 65
RATE_LIMIT_RETRIES    = 5
SERVICE_ERROR_RETRIES = 5

WORD_LIMIT = 4000

TARGET_REPOS = {
    "pallets/flask",
    "sphinx-doc/sphinx",
    "pytest-dev/pytest",
    "psf/requests",
    "mwaskom/seaborn",
    "pylint-dev/pylint",
    "sympy/sympy",
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "pydata/xarray",
}

REPO_PATHS = {
    "pallets/flask":           Path("repos/pallets_flask"),
    "pytest-dev/pytest":       Path("repos/pytest-dev_pytest"),
    "sphinx-doc/sphinx":       Path("repos/sphinx-doc_sphinx"),
    "psf/requests":            Path("repos/psf_requests"),
    "mwaskom/seaborn":         Path("repos/mwaskom_seaborn"),
    "pylint-dev/pylint":       Path("repos/pylint-dev_pylint"),
    "sympy/sympy":             Path("repos/sympy_sympy"),
    "astropy/astropy":         Path("repos/astropy_astropy"),
    "django/django":           Path("repos/django_django"),
    "matplotlib/matplotlib":   Path("repos/matplotlib_matplotlib"),
    "scikit-learn/scikit-learn": Path("repos/scikit-learn_scikit-learn"),
    "pydata/xarray":           Path("repos/pydata_xarray"),
}

RESULTS_FILE = {
    "ecocode":  Path("results_ecocode.json"),
    "baseline": Path("results_baseline.json"),
}


def _load_results_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _upsert_result(accum: list, result: dict) -> None:
    """Replace row with same instance_id or append (merge across runs)."""
    iid = result["instance_id"]
    for i, row in enumerate(accum):
        if isinstance(row, dict) and row.get("instance_id") == iid:
            accum[i] = result
            return
    accum.append(result)


_SYSTEM_PROMPT = (
    "You are a surgical code repair assistant. "
    "You will be given a Python source file and a bug description. "
    "Your ONLY job is to fix the specific bug — nothing else.\n\n"
    "STRICT RULES:\n"
    "1. Output the COMPLETE source file with the minimal change applied.\n"
    "2. Every line that is not part of the fix must be copied EXACTLY — "
    "same whitespace, same wording, same line breaks. Do not touch anything else.\n"
    "3. Make the FEWEST possible line changes. If 2 lines fix the bug, change only those 2 lines.\n"
    "4. Do NOT add, remove, or modify any comments or docstrings.\n"
    "5. Do NOT rename variables, reformat code, or reorganize logic.\n"
    "6. Do NOT wrap output in markdown fences, backticks, or any other formatting.\n"
    "7. Do NOT write any explanation before or after the code.\n"
    "Your entire response must be the raw source file and nothing else."
)

SYSTEM_PROMPTS = {
    "gemini":     _SYSTEM_PROMPT,
    "cerebras":   _SYSTEM_PROMPT,
    "groq":       _SYSTEM_PROMPT,
    "openrouter": _SYSTEM_PROMPT,
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_patch_f1(original: str, llm_output: str, gold_patch: str) -> float:
    """Patch F1: precision/recall on the set of lines added by LLM vs gold patch.

    gold_adds: lines the gold patch adds (from row["patch"])
    llm_adds:  lines the LLM added vs the original (via difflib)
    Precision: fraction of LLM additions that match gold (penalises spurious changes)
    Recall:    fraction of gold additions reproduced by LLM (penalises missing fix)
    Returns 0.0 if either set is empty.
    """
    gold_adds = {
        l[1:] for l in gold_patch.splitlines()
        if l.startswith("+") and not l.startswith("+++")
    }
    llm_diff = difflib.unified_diff(
        original.splitlines(), llm_output.splitlines(), lineterm=""
    )
    llm_adds = {
        l[1:] for l in llm_diff
        if l.startswith("+") and not l.startswith("+++")
    }

    if not gold_adds or not llm_adds:
        return 0.0

    inter     = gold_adds & llm_adds
    precision = len(inter) / len(llm_adds)
    recall    = len(inter) / len(gold_adds)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_hunk_bleu(llm_output: str, gold_patch: str) -> float:
    """Hunk BLEU: sacrebleu score on the lines touched by each gold hunk.

    For each hunk, extract the same target line range from the LLM output
    and compute sentence BLEU against the gold lines. Returns the average
    BLEU across all hunks (0–100). Returns 0.0 on parse error.
    """
    try:
        patch     = PatchSet(gold_patch)
        llm_lines = llm_output.splitlines()
        scores    = []

        for patched_file in patch:
            for hunk in patched_file:
                gold_lines = [l.value.rstrip("\n") for l in hunk if l.is_added]
                if not gold_lines:
                    continue
                start    = hunk.target_start - 1
                llm_hunk = llm_lines[start : start + hunk.target_length]
                score    = sacrebleu.sentence_bleu(
                    " ".join(llm_hunk),
                    [" ".join(gold_lines)],
                ).score
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0
    except Exception as e:
        print(f"  [bleu] parse error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def build_prompt(problem_statement: str, test_patch: str, filepath: str, file_content: str) -> str:
    return (
        f"FILE TO FIX: {filepath}\n\n"
        f"BUG DESCRIPTION:\n{problem_statement}\n\n"
        f"TESTS THAT MUST PASS (shows the exact API/behavior required):\n{test_patch}\n\n"
        f"INSTRUCTION: Apply the minimal fix to make those tests pass. "
        f"Copy every other line exactly as-is. Do not add comments. "
        f"Output the complete raw file only.\n\n"
        f"CURRENT FILE CONTENT:\n{file_content}"
    )


async def call_llm(model: str, prompt: str) -> tuple[str, int]:
    """Call LLM with rate-limit and service-error retry logic.
    Returns (text, total_tokens) where total_tokens comes from response.usage."""
    if model.startswith("openrouter/"):
        provider = "openrouter"
    elif "groq/" in model:
        provider = "groq"
    elif "cerebras/" in model:
        provider = "cerebras"
    else:
        provider = "gemini"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[provider]},
        {"role": "user",   "content": prompt},
    ]

    extra = {}
    if provider == "openrouter":
        k = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if k:
            extra["api_key"] = k

    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            response = await litellm.acompletion(model=model, messages=messages, **extra)
            text   = response.choices[0].message.content or ""
            tokens = getattr(response.usage, "total_tokens", 0) or 0
            return text, tokens
        except litellm.exceptions.RateLimitError as e:
            wait = _parse_retry_after(str(e)) + 2
            print(f"  [rate limit] waiting {wait}s (attempt {attempt})...", flush=True)
            await asyncio.sleep(wait)
        except litellm.exceptions.ServiceUnavailableError:
            wait = 30 * attempt
            print(f"  [503] waiting {wait}s (attempt {attempt})...", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            print(f"  [LLM error] {e}")
            return "", 0

    print("  [LLM] max retries exceeded, skipping.")
    return "", 0


def _parse_retry_after(error_msg: str) -> int:
    """Parse retry-after seconds from a rate-limit error message.
    Handles 'N seconds' and 'Nms' formats (ms rounded up to 1s minimum)."""
    m_sec = re.search(r"(\d+)\s*second", error_msg)
    if m_sec:
        return int(m_sec.group(1))
    m_ms = re.search(r"(\d+)\s*ms", error_msg)
    if m_ms:
        return max(1, int(m_ms.group(1)) // 1000)
    return 60


def clean_llm_output(raw: str) -> str:
    """Strip accidental markdown fences if the model ignored instructions."""
    stripped = raw.strip()
    m = re.match(r"^```[^\n]*\n(.*?)\n```\s*$", stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Opening ``` / ```python line without a closing fence (common with Groq)
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        stripped = stripped[nl + 1 :] if nl != -1 else stripped.lstrip("`")
    # Stray closing fence at end
    stripped = re.sub(r"\n?```\s*$", "", stripped.rstrip())
    return stripped.strip()


# ---------------------------------------------------------------------------
# Diff preview
# ---------------------------------------------------------------------------

def _print_diff_preview(original: str, llm_output: str, filepath: str, context: int = 3) -> None:
    """Print a unified diff of what the LLM changed, capped at 60 lines.
    This lets you verify the model only touched what it needed to."""
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        llm_output.splitlines(keepends=True),
        fromfile=f"original/{filepath}",
        tofile=f"llm/{filepath}",
        n=context,
    ))
    if not diff_lines:
        print("  [diff] no changes made")
        return

    added   = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    print(f"  [diff] +{added} lines  -{removed} lines")

    cap = 60
    for line in diff_lines[:cap]:
        print("  " + line, end="")
    if len(diff_lines) > cap:
        print(f"\n  ... ({len(diff_lines) - cap} more diff lines not shown)")
    print()


# ---------------------------------------------------------------------------
# Core per-example logic
# ---------------------------------------------------------------------------

async def run_example(row: dict, mode: str) -> dict | None:
    """Process one SWE-bench example. Returns a result dict or None if skipped.

    The LLM output stays in memory only — nothing is ever written to disk,
    so no git restore is needed after each example.
    """
    instance_id       = row["instance_id"]
    repo              = row["repo"]
    base_commit       = row["base_commit"]
    problem_statement = row["problem_statement"]
    test_patch        = row["test_patch"]
    gold_patch        = row["patch"]

    # Step 1: find target file from gold patch diff header (single-file only;
    # metrics assume one file matches the LLM full-file output).
    patch_files = re.findall(r"^diff --git a/(.*?) b/", gold_patch, re.MULTILINE)
    if not patch_files:
        print("  [skip] no target file found in patch")
        return None
    unique_patch_files = list(dict.fromkeys(patch_files))
    if len(unique_patch_files) != 1:
        print(f"  [skip] multi-file gold patch ({len(unique_patch_files)} files)")
        return None
    target_file_rel = unique_patch_files[0]

    # Step 2: checkout base_commit and read original file
    repo_path   = repo_checker(repo, base_commit)
    target_path = repo_path / target_file_rel
    if not target_path.exists():
        print(f"  [skip] target file not found: {target_file_rel}")
        return None
    original_content = load_full_file(target_path)

    # Step 3: skip if file too large
    word_count = len(original_content.split())
    if word_count > WORD_LIMIT:
        print(f"  [skip] file too large ({word_count} words)")
        return None

    # Step 4: determine model and energy
    input_data = InputData(
        instance_id=instance_id,
        problem_statement=problem_statement,
        repo=repo,
        base_commit=base_commit,
        resolved_path=str(target_path),
        file_content=original_content,
    )

    if mode == "ecocode":
        routing   = route_input(input_data)
        model_key = routing.model
        model     = TIER_TO_MODEL[model_key]
    else:
        model_key = LARGE_MODEL
        model     = BASELINE_MODEL

    # Step 5: call LLM — output stays in memory, never written to disk
    sleep_s = INTER_CALL_SLEEP_GROQ if "groq/" in model else INTER_CALL_SLEEP
    print(f"  sleeping {sleep_s}s...", end=" ", flush=True)
    await asyncio.sleep(sleep_s)
    print("calling LLM...", flush=True)

    prompt              = build_prompt(problem_statement, test_patch, target_file_rel, original_content)
    raw_output, tokens  = await call_llm(model, prompt)

    # Energy = per-token rate (kWh/token) × actual tokens used (from API response)
    energy = tokens * MODEL_ENERGY_COST[model_key]

    if not raw_output:
        return {
            "instance_id": instance_id, "model_key": model_key,
            "tokens": tokens, "energy": energy,
            "bleu": 0.0, "f1": 0.0, "exact_match": False, "status": "llm_error",
        }

    llm_output = clean_llm_output(raw_output)

    if llm_output == original_content:
        print("  [warn] LLM output identical to original")

    # Show a compact diff so we can verify the LLM only changed what it needed to
    _print_diff_preview(original_content, llm_output, target_file_rel)

    # Step 6: compute metrics in memory
    f1          = compute_patch_f1(original_content, llm_output, gold_patch)
    bleu        = compute_hunk_bleu(llm_output, gold_patch)
    exact_match = f1 == 1.0

    return {
        "instance_id": instance_id,
        "model_key":   model_key,
        "tokens":      tokens,
        "energy":      energy,
        "bleu":        round(bleu, 2),
        "f1":          round(f1, 4),
        "exact_match": exact_match,
        "status":      "exact" if exact_match else ("partial" if f1 > 0 else "failed"),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(mode: str, n: int | None, instance: str | None) -> None:
    from datasets import load_dataset

    print(f"\n{'='*52}")
    print(f"  EcoCode Evaluation — mode={mode}")
    print(f"{'='*52}\n")

    ds       = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    examples = [r for r in ds if r["repo"] in TARGET_REPOS]

    if instance:
        examples = [r for r in examples if r["instance_id"] == instance]
        if not examples:
            print(f"Instance '{instance}' not found.")
            return
    results_path = RESULTS_FILE[mode]
    merged       = _load_results_list(results_path)
    session      = []
    limit        = n  # stop after this many *processed* (non-skipped) examples this run

    for idx, row in enumerate(examples, 1):
        if limit and len(session) >= limit:
            break
        print(f"[{idx}] {row['instance_id']}  (processed {len(session)}/{limit or '∞'})")
        result = await run_example(row, mode)

        if result is None:
            print("  → skipped\n")
            continue

        _upsert_result(merged, result)
        session.append(result)
        results_path.write_text(json.dumps(merged, indent=2))
        print(
            f"  → model={result['model_key']}  "
            f"bleu={result['bleu']:.1f}  f1={result['f1']:.3f}  "
            f"exact={result['exact_match']}  "
            f"energy={result['energy']:.6f} kWh"
            f"\n  → appended to {results_path} ({len(merged)} total rows)\n"
        )

    try:
        from litellm import close_litellm_async_clients

        await close_litellm_async_clients()
    except Exception:
        pass

    if session:
        print(f"Results file: {results_path} ({len(merged)} rows)")

    n_ran        = len(session)
    n_exact      = sum(1 for r in session if r["exact_match"])
    avg_bleu     = sum(r["bleu"] for r in session) / n_ran if n_ran else 0
    avg_f1       = sum(r["f1"]   for r in session) / n_ran if n_ran else 0
    total_energy = sum(r["energy"] for r in session)

    print(f"\nSUMMARY ({mode}) — this run only")
    if n_ran:
        print(f"  Examples:     {n_ran}")
        print(f"  Avg BLEU:     {avg_bleu:.1f}")
        print(f"  Avg F1:       {avg_f1:.3f}")
        print(f"  Exact Match:  {n_exact}/{n_ran}  ({100*n_exact/n_ran:.1f}%)")
        print(f"  Total energy: {total_energy:.6f} kWh")
    else:
        print("  No new results this run.")


# ---------------------------------------------------------------------------
# Compare results
# ---------------------------------------------------------------------------

def compare_results() -> None:
    eco_path  = RESULTS_FILE["ecocode"]
    base_path = RESULTS_FILE["baseline"]

    if not eco_path.exists() or not base_path.exists():
        print("Run both --mode ecocode and --mode baseline first.")
        return

    eco  = json.loads(eco_path.read_text())
    base = json.loads(base_path.read_text())

    def stats(results):
        n        = len(results)
        n_exact  = sum(1 for r in results if r["exact_match"])
        avg_bleu = sum(r["bleu"] for r in results) / n if n else 0
        avg_f1   = sum(r["f1"]   for r in results) / n if n else 0
        energy   = sum(r["energy"] for r in results)
        return n, n_exact, avg_bleu, avg_f1, energy

    e_n, e_exact, e_bleu, e_f1, e_energy = stats(eco)
    b_n, b_exact, b_bleu, b_f1, b_energy = stats(base)
    savings = (1 - e_energy / b_energy) * 100 if b_energy else 0

    print(f"\n{'='*64}")
    print(f"  EcoCode vs Baseline — SWE-bench Lite Comparison")
    print(f"{'='*64}")
    print(f"  {'Mode':<12} {'BLEU':>6}  {'F1':>6}  {'ExactMatch':>12}  {'Energy(kWh)':>12}  {'Savings':>8}")
    print(f"  {'-'*60}")
    print(f"  {'ecocode':<12} {e_bleu:>6.1f}  {e_f1:>6.3f}  {f'{e_exact}/{e_n}':>12}  {e_energy:>12.6f}  {savings:>7.1f}%")
    print(f"  {'baseline':<12} {b_bleu:>6.1f}  {b_f1:>6.3f}  {f'{b_exact}/{b_n}':>12}  {b_energy:>12.6f}  {'—':>8}")
    print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------

def run_tests_sanity() -> None:
    print("\nRunning sanity checks...\n")

    # Test 1: repos exist
    print("1. Checking repos exist...")
    all_ok = True
    for repo, path in REPO_PATHS.items():
        exists = path.exists()
        print(f"   {path}: {'OK' if exists else 'MISSING'}")
        if not exists:
            all_ok = False
    print("   " + ("PASS" if all_ok else "FAIL — repos clone on first evaluation run"))

    # Test 2: compute_patch_f1 with known values
    print("\n2. Checking compute_patch_f1...")
    original = "def foo():\n    return 1\n"
    fixed    = "def foo():\n    return 2\n"
    patch    = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n+++ b/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def foo():\n-    return 1\n+    return 2\n"
    )
    f1_perfect = compute_patch_f1(original, fixed, patch)
    f1_noop    = compute_patch_f1(original, original, patch)
    print(f"   perfect fix → F1={f1_perfect:.3f}  {'PASS' if f1_perfect == 1.0 else 'FAIL'}")
    print(f"   no-op fix   → F1={f1_noop:.3f}   {'PASS' if f1_noop == 0.0 else 'FAIL'}")

    # Test 3: compute_hunk_bleu returns float in range
    print("\n3. Checking compute_hunk_bleu...")
    bleu = compute_hunk_bleu(fixed, patch)
    ok   = isinstance(bleu, float) and 0.0 <= bleu <= 100.0
    print(f"   bleu={bleu:.1f}  {'PASS' if ok else 'FAIL'}")

    # Test 4: clean_llm_output strips partial markdown fences
    print("\n4. Checking clean_llm_output...")
    partial = "```python\nimport x\n"
    cleaned = clean_llm_output(partial)
    ok_fence = cleaned == "import x"
    print(f"   partial fence → {repr(cleaned)}  {'PASS' if ok_fence else 'FAIL'}")

    print("\nDone.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EcoCode SWE-bench evaluation")
    parser.add_argument("--mode", choices=["ecocode", "baseline"],
                        help="Which mode to run")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of examples to run (default: all)")
    parser.add_argument("--instance", type=str, default=None,
                        help="Run a specific instance by ID")
    parser.add_argument("--compare", action="store_true",
                        help="Compare saved ecocode vs baseline result JSONs")
    parser.add_argument("--test", action="store_true",
                        help="Run sanity checks")
    args = parser.parse_args()

    if args.compare:
        compare_results()
        return

    if args.test:
        run_tests_sanity()
        return

    if not args.mode:
        parser.error("--mode is required (ecocode or baseline)")

    asyncio.run(run_evaluation(args.mode, args.n, args.instance))


if __name__ == "__main__":
    main()
