#!/usr/bin/env python3
"""Stats: easy / medium / hard for rows evaluate.py would run (same skip rules).

Iterates in the same order as evaluate.py: Lite test split, filtered by TARGET_REPOS,
preserving HuggingFace row order (not sorted by repo).

  python complexity_stats.py
  python complexity_stats.py --attempts 50
      First 50 dataset rows only. Skips (e.g. file too large) still count as rows
      in that window — matches "looked at 50 rows, 39 under WORD_LIMIT" style stats.

  python complexity_stats.py --processed 50
      Same semantics as evaluate.py --n 50: walk from the start until 50 rows would
      actually run (each non-skip adds to the bucket), skipping large files etc.

Skip rules match evaluate.run_example. WORD_LIMIT is word count on the target file.
Complexity: logic.complexity_assessment (prompt words + 0.3 * file words vs 400 / 800).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from evaluate import TARGET_REPOS, WORD_LIMIT
from logic import (
    REPO_DIR,
    InputData,
    complexity_assessment,
    load_full_file,
    run_git,
)


def target_file_from_patch(gold_patch: str) -> str | None:
    patch_files = re.findall(r"^diff --git a/(.*?) b/", gold_patch, re.MULTILINE)
    if not patch_files:
        return None
    unique = list(dict.fromkeys(patch_files))
    if len(unique) != 1:
        return None
    return unique[0]


def ensure_repo_fetched(repo_name: str) -> Path:
    owner, name = repo_name.split("/")
    repo_path = REPO_DIR / f"{owner}_{name}"
    remote_url = f"https://github.com/{owner}/{name}.git"
    if not repo_path.exists():
        run_git(["clone", remote_url, str(repo_path)])
    run_git(["fetch", "--all", "--tags"], cwd=repo_path)
    return repo_path


def checkout(repo_path: Path, base_commit: str) -> None:
    run_git(["checkout", "-f", base_commit], cwd=repo_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="EcoCode complexity bucket stats for eval slice")
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--attempts",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N dataset rows from the start (default: no limit)",
    )
    g.add_argument(
        "--processed",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N runnable rows (same idea as evaluate.py --n N)",
    )
    args = parser.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    rows_all = [r for r in ds if r["repo"] in TARGET_REPOS]
    total_in_slice = len(rows_all)

    skip = Counter()
    bucket = Counter()
    repo_paths: dict[str, Path] = {}
    rows_visited = 0
    runnable = 0

    for row in rows_all:
        if args.processed is not None and runnable >= args.processed:
            break
        if args.attempts is not None and rows_visited >= args.attempts:
            break

        rows_visited += 1
        repo_name = row["repo"]

        if repo_name not in repo_paths:
            try:
                repo_paths[repo_name] = ensure_repo_fetched(repo_name)
            except Exception:
                skip["repo_fetch_error"] += 1
                continue

        repo_path = repo_paths[repo_name]
        gold = row["patch"]
        rel = target_file_from_patch(gold)
        if rel is None:
            skip["bad_or_multi_file_patch"] += 1
            continue

        try:
            checkout(repo_path, row["base_commit"])
        except Exception:
            skip["repo_checkout_error"] += 1
            continue

        path = repo_path / rel
        if not path.exists():
            skip["target_missing"] += 1
            continue

        content = load_full_file(path)
        words = len(content.split())
        if words > WORD_LIMIT:
            skip["file_too_large"] += 1
            continue

        inp = InputData(
            instance_id=row["instance_id"],
            problem_statement=row.get("problem_statement", ""),
            repo=repo_name,
            base_commit=row["base_commit"],
            resolved_path=str(path),
            file_content=content,
        )
        bucket[complexity_assessment(inp)] += 1
        runnable += 1

    n_run = sum(bucket.values())
    n_skip = sum(skip.values())
    print(f"TARGET_REPOS: {len(TARGET_REPOS)} repos, WORD_LIMIT={WORD_LIMIT} words")
    print(f"Rows in full slice: {total_in_slice}")
    if args.attempts is not None:
        print(f"Mode: --attempts {args.attempts} (visited {rows_visited} dataset row(s) in eval order)")
    elif args.processed is not None:
        print(f"Mode: --processed {args.processed} (visited {rows_visited} row(s) to get {n_run} runnable)")
    else:
        print(f"Mode: full slice (visited {rows_visited} row(s))")
    print(f"Would run (no skip):   {n_run}")
    print(f"Skipped (in window):   {n_skip}")
    for k, v in sorted(skip.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {v:4d}  {k}")
    print()
    if n_run:
        for label in ("easy", "medium", "hard"):
            c = bucket[label]
            print(f"  {label:6s}  {c:4d}  ({100.0 * c / n_run:.1f}%)")
    else:
        print("  (no runnable rows in window)")
    sys.exit(0)


if __name__ == "__main__":
    main()
