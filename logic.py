from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from xml.parsers.expat import model
import argparse
import re
import subprocess
import os

from pathlib import Path


SMALL_MODEL = "small"
MEDIUM_MODEL = "medium"
LARGE_MODEL = "large"

MAPPING_TO_MODEL = {
    SMALL_MODEL:  "Gemini 2.5 Flash",
    MEDIUM_MODEL: "Groq LLaMA-3.3 70B",
    LARGE_MODEL:  "Cerebras Qwen-3 235B",
}

# Energy cost in kWh per token, derived from published per-query measurements.
# All Wh/query figures divided by total tokens in that query configuration.
#
#   small  (Gemini Flash):
#     Google official: 0.24 Wh per median text prompt (~600 tokens)
#     → 0.24 Wh / 600 tokens = 4.0e-7 kWh/token
#     Source: Google Environmental Report (2025), via ByteThirst v2.0
#
#   medium (Cerebras gpt-oss-120b):
#     Best available proxy: LLaMA-3.3-70B on AWS H200 (Table 4, Jegham et al. 2025)
#     Short query (100 input + 300 output = 400 tokens): 0.237 ± 0.023 Wh
#     → 0.237 Wh / 400 tokens = 5.9e-7 kWh/token
#     Note: 120B > 70B so this is a slight underestimate; treated as conservative lower bound.
#     Source: Jegham et al. arXiv:2505.09598 (2025), Table 4
#
#   large  (Cerebras qwen-3-235b):
#     Linear interpolation between same-generation, same-architecture pure-text models
#     from Table 4, Jegham et al. (2025), both hosted on AWS H200/H100:
#       LLaMA-3.1-70B:  1.271 Wh / 400 tok = 3.18e-6 kWh/token
#       LLaMA-3.1-405B: 2.226 Wh / 400 tok = 5.57e-6 kWh/token
#     235B is (235-70)/(405-70) = 49.3% of the way from 70B to 405B
#     → 1.271 + 0.493 × (2.226 - 1.271) = 1.742 Wh / 400 tok = 4.35e-6 kWh/token
#     Source: Jegham et al. arXiv:2505.09598 (2025), Table 4
MODEL_ENERGY_COST = {
    SMALL_MODEL:  4.0e-7,
    MEDIUM_MODEL: 5.9e-7,
    LARGE_MODEL:  4.35e-6,
}

THRESHOLDS = {
    "easy": 400.0,
    "medium": 800.0,
}

REPO_DIR = Path("repos")
REPO_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATTERN = re.compile(
    r"\b(?:[\w\-/]+/)?[\w\-]+\.(?:py|pyi|js|ts|java|cpp|c|h|go|rb|php|rs)\b"
)


@dataclass
class InputData:
    instance_id: str
    problem_statement: str
    repo: str
    base_commit: str
    resolved_path: Optional[str] = None
    mentioned_file_hint: Optional[str] = None
    file_content: str = ""

    @property
    def prompt_token_count(self) -> int:
        return len(self.problem_statement.split())
    
    @property
    def file_token_count(self) -> int:
        return len(self.file_content.split())
    
    @property
    def total_token_count(self) -> int:
        return self.prompt_token_count + 0.3 * self.file_token_count

    
@dataclass
class OutputData:
    complexity: str
    model: str
    model_name: str
    estimated_saved_energy: float
    reasoning: str
    savings_percentage: float


def run_git(args: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(["git"] + args, cwd=str(cwd) if cwd else None, check=True, 
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
def repo_checker(repo_name: str, base_commit: str) -> Path:
    owner, name = repo_name.split("/")
    repo_path = REPO_DIR / f"{owner}_{name}"   
    remote_url = f"https://github.com/{owner}/{name}.git"

    if not repo_path.exists():
        run_git(["clone", remote_url, str(repo_path)])

    run_git(["fetch", "--all", "--tags"], cwd=repo_path)
    run_git(["checkout", "-f", base_commit], cwd=repo_path)
    return repo_path

def extract_file(problem_statement: str) -> List[str]:
    matches = [m.group(0) for m in FILE_PATTERN.finditer(problem_statement or "")]
    seen = set()
    ordered_unique = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            ordered_unique.append(match)
    return ordered_unique

def resolved_file_path(repo_dir: Path, file_hint: str) -> Optional[Path]:
    normalized_hint = file_hint.replace("\\", "/").strip().lower()
    repo_files = [
        p for p in repo_dir.rglob("*")
        if p.is_file() and ".git" not in p.parts
    ]
    for p in repo_files:
        rel = str(p.relative_to(repo_dir)).replace("\\", "/").lower()
        if rel == normalized_hint:
            return p
    for p in repo_files:
        rel = str(p.relative_to(repo_dir)).replace("\\", "/").lower()
        if rel.endswith(normalized_hint):
            return p
    hint_basename = Path(normalized_hint).name
    basename_matches = [p for p in repo_files if p.name.lower() == hint_basename]

    if len(basename_matches) == 1:
        return basename_matches[0]

    if basename_matches:
        basename_matches.sort(key=lambda p: len(str(p.relative_to(repo_dir))))
        return basename_matches[0]

    return None

def load_full_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def converter(row: dict) -> InputData:
    problem_statement = row.get("problem_statement", "")
    mentioned_files = extract_file(problem_statement)
    if not mentioned_files:
        return None

    return InputData(
        instance_id = row["instance_id"],
        problem_statement=problem_statement,
        repo=row["repo"],
        base_commit=row["base_commit"],
        mentioned_file_hint=mentioned_files[0],
    )

def prepare_input_data(row: dict) -> Optional[InputData]:
    input_data = converter(row)
    if input_data is None:
        return None
    
    repo_dir = repo_checker(input_data.repo, input_data.base_commit)
    resolved_path = resolved_file_path(repo_dir, input_data.mentioned_file_hint)

    if resolved_path is None:
        return None
    
    input_data.resolved_path = str(resolved_path)
    input_data.file_content = load_full_file(resolved_path)
    return input_data

def complexity_assessment(input_data: InputData) -> str:
    total_len = input_data.total_token_count
    if total_len < THRESHOLDS["easy"]:
        complexity = "easy"
    elif total_len < THRESHOLDS["medium"]:
        complexity = "medium"
    else:
        complexity = "hard"

    return complexity
    
def model_selection(input_data: InputData) -> str:
    complexity = complexity_assessment(input_data)
    if complexity == "easy":
        return SMALL_MODEL
    elif complexity == "medium":
        return MEDIUM_MODEL
    else:
        return LARGE_MODEL
    
    
def energy_savings_estimation(input_data: InputData, model: str) -> Tuple[float, float]:
    tokens = input_data.total_token_count
    baseline_energy = tokens * MODEL_ENERGY_COST[LARGE_MODEL]
    selected_energy = tokens * MODEL_ENERGY_COST[model]

    estimated_saved_energy = max(0.0, baseline_energy - selected_energy)
    savings_percentage = (estimated_saved_energy / baseline_energy) * 100

    return estimated_saved_energy, savings_percentage

def reasoning_explanation(input_data: InputData, model: str) -> str:
    complexity = complexity_assessment(input_data)
    suggested_model_energy = MODEL_ENERGY_COST[model]
    large_model_energy = MODEL_ENERGY_COST[LARGE_MODEL]
    savings = large_model_energy - suggested_model_energy
    savings_percentage = (MODEL_ENERGY_COST[LARGE_MODEL] - MODEL_ENERGY_COST[model]) / MODEL_ENERGY_COST[LARGE_MODEL] * 100


    reasoning = f"The input prompt contains {input_data.total_token_count:.1f} tokens."
    reasoning += f" it classified as {complexity} complexity."
    reasoning += f" Considering the complexity, the suggested model is {model} which has an energy cost of {suggested_model_energy} units."
    reasoning += f" By choosing this model, compared to the baseline of {LARGE_MODEL} with an energy cost of {large_model_energy} units, the estimated savings is {savings} units; therefore, approximately {savings_percentage:.2f}% savings."


    return reasoning


def route_input(input_data: InputData) -> OutputData:
    model = model_selection(input_data)
    model_name = MAPPING_TO_MODEL[model]
    estimated_saved_energy, savings_percentage = energy_savings_estimation(input_data, model)
    reasoning = reasoning_explanation(input_data, model)
    
    return OutputData(complexity=complexity_assessment(input_data), 
                      model=model, 
                      model_name=model_name, 
                      estimated_saved_energy=estimated_saved_energy, 
                      reasoning=reasoning, 
                      savings_percentage=savings_percentage)


def main_swebench(row: dict) -> Optional[OutputData]:
    input_data = prepare_input_data(row)
    if input_data is None:
        return None
    return route_input(input_data)


def manage_input(prompt):
    ##testing case
    if "--test" in prompt:
        return {
            "complexity": "Simple",
            "model": "Small",
            "model_key": "small",
            "energy": 0.0012,
            "baseline": 0.0058,
            "savings": 79,
            "tokens": 320,
            "context_files": [],
            "reasoning": "some reasoning here",
        }
    
    potential_paths = re.findall(r'(\/[^\n,]+?\.[a-zA-Z0-9]+)', prompt)
    files = [p.strip(',.?!') for p in potential_paths if os.path.exists(p.strip(',.?!'))]

    file_content = ""
    resolved_path = None

    if files:
        resolved_path = files[0]
        file_content = Path(resolved_path).read_text(encoding="utf-8", errors="ignore")

    input_data = InputData(
        instance_id="cli",
        repo = "local",
        base_commit="local",
        problem_statement=prompt,
        mentioned_file_hint=resolved_path,
        resolved_path=resolved_path,
        file_content=file_content
    )   
    output = route_input(input_data)
    baseline_energy =  input_data.total_token_count * MODEL_ENERGY_COST[LARGE_MODEL]
    energy = input_data.total_token_count * MODEL_ENERGY_COST[output.model]


    return {
            "complexity": output.complexity,
            "model": output.model_name,
            "model_key": output.model,
            "energy": energy,
            "baseline": baseline_energy,
            "savings": output.savings_percentage,
            "tokens": input_data.total_token_count,
            "context_files": files,
            "reasoning": output.reasoning,
        }

def evaluate(n: int = 100) -> None:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite")
    kept = 0

    for row in ds["test"]:
        if kept >= n:
            break

        output = main_swebench(row)
        if output is None:
            continue

        kept += 1
        print(f"Example {kept}:")
        print("Instance ID:", row["instance_id"])
        print("Repo:", row["repo"])
        print("Problem Statement:", row.get("problem_statement", "")[:500])
        print("Model:", output.model)
        print("Complexity:", output.complexity)
        print("Estimated Saved Energy:", output.estimated_saved_energy)
        print("Savings Percentage:", f"{output.savings_percentage:.2f}%")
        print("Reasoning:", output.reasoning)
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="EcoRoute CLI")
    parser.add_argument("prompt", nargs="?", type=str)

    args = parser.parse_args()

    result = manage_input(args.prompt)

    print("\n" + "="*40)
    print("    ECOCODE SUSTAINABILITY DASHBOARD")
    print("="*40)
    print(f"TASK DETECTED:    {args.prompt[:30]}...")
    if result['context_files']:
        print(f"CONTEXT AUDIT:    Found file(s): {result['context_files']}")
    else:
        print(f"CONTEXT AUDIT:    NO FILE FOUND")
    print("-" * 40)
    print(f"COMPLEXITY:       {result['complexity']}")
    print(f"RECOMMENDED:      {result['model']}")
    print("-" * 40)
    print(f"EST. ENERGY:      {result['energy']:.6f}".rstrip('0').rstrip('.') + " kWh")
    print(f"SAVINGS:          {result['savings']:.2f}% ")
    print(f"EST. TOKENS:      {result['tokens']}")
    print(f"REASONING:        {result['reasoning']}")
    print("="*40 + "\n")



if __name__ == "__main__":
    main()