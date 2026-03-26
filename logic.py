from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

ds = load_dataset("SWE-bench/SWE-bench_Verified")


SMALL_MODEL = "small"
MEDIUM_MODEL = "medium"
LARGE_MODEL = "large"

MAPPING_TO_MODEL = {
    SMALL_MODEL: "tbd_small",
    MEDIUM_MODEL: "tbd_medium",
    LARGE_MODEL: "tbd_large",
}

MODEL_ENERGY_COST = {
    SMALL_MODEL: 0.1,
    MEDIUM_MODEL: 0.5,
    LARGE_MODEL: 1.0,
}

@dataclass
class InputData:
    problem_statement: str
    hints_text: str = ""
    file_paths: List[str] = field(default_factory=list)
    swe_bench_difficulty: Optional[str] = None

    @property
    def num_files(self) -> int:
        return max(1, len(self.file_paths))
    
    @property
    def includes_hints(self) -> bool:
        return bool(self.hints_text and self.hints_text.strip())
    
    @property
    def combined(self) -> str:
        return f"{self.problem_statement}\n{self.hints_text}".strip()
    
    @property
    def token_count(self) -> int:
        return len(self.combined.split())
    
@dataclass
class OutputData:
    intent: str
    complexity: str
    model: str
    model_name: str
    estimated_saved_energy: float
    reasoning: str
    savings_percentage: float
    swe_bench_difficulty: Optional[str] = None

def converter(row: dict) -> InputData:
    return InputData(
        problem_statement=row.get("problem_statement", ""),
        hints_text= row.get("hints_text", "") or "",
        file_paths=[],
        swe_bench_difficulty=row.get("difficulty"),
    )

def intent_detection(input_data: InputData) -> str:
    problem_statement = input_data.problem_statement.lower()

    keywords = {
        # hard
        "code_generation": ["write code", "generate code", "create code", "implement"],
        # medium
        "code_explanation": ["explain code", "describe code", "code explanation", "code walkthrough"],
        # medium
        "bug_fixing": ["fix bug", "debug code", "troubleshoot", "resolve issue", "indexerror", "valueerror", "attributeerror", "does not", "incorrect", "wrong", "fails", "failing", "unexpected", "broken", "not working", "cannot", "crash", "exception"],
        # medium
        "general_query": ["what is", "how to", "why does", "explain", "describe", "tell me about"],
        # easy
        "simple_edit": ["change", "modify", "update", "refactor", "format", "typo"],
        # medium
        "exploratory_analysis": ["analyze", "explore", "data analysis", "visualize data"],
        # hard
        "feature_request": ["add feature", "new feature", "feature request", "enhancement", "enable"],
    }

    for intent, key_phrases in keywords.items():
        if any(phrase in problem_statement for phrase in key_phrases):
            return intent
    return "unknown_intent"

def complexity_assessment(input_data: InputData) -> str:
    token_count = input_data.token_count
    num_files = input_data.num_files
    intent = intent_detection(input_data)
    includes_hints = input_data.includes_hints

    if token_count < 250 and num_files <= 1 and intent in ["simple_edit"]:
        complexity = "easy"
    elif token_count < 1200 and num_files <= 3 and intent in ["code_explanation", "bug_fixing", "general_query", "exploratory_analysis"]:
        complexity = "medium"
    # if there is unknown intent classify it as medium 
    elif token_count < 800 and num_files <=2 and intent in ["unknown_intent"]:
        complexity = "medium"
    else:
        complexity = "hard"
    
    # also including hints in the calculation
    # if we have hints, we can change hard to medium
    if complexity == "hard" and includes_hints and token_count < 1500 and num_files <= 2:
        complexity = "medium"

  

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
    complexity = complexity_assessment(input_data)
    baseline_energy = MODEL_ENERGY_COST[LARGE_MODEL]
    if complexity == "easy":
        selected_energy = MODEL_ENERGY_COST[model]
    elif complexity == "medium":
        selected_energy = MODEL_ENERGY_COST[model]
    else:
        selected_energy = MODEL_ENERGY_COST[model]

    estimated_saved_energy = max(0.0, baseline_energy - selected_energy)
    savings_percentage = (estimated_saved_energy / baseline_energy) * 100 if baseline_energy > 0 else 0

    return estimated_saved_energy, savings_percentage

def reasoning_explanation(input_data: InputData, model: str) -> str:
    complexity = complexity_assessment(input_data)
    intent = intent_detection(input_data)
    token_num = input_data.token_count
    number_of_files = input_data.num_files
    suggested_model_energy = MODEL_ENERGY_COST[model]
    large_model_energy = MODEL_ENERGY_COST[LARGE_MODEL]
    savings = large_model_energy - suggested_model_energy
    savings_percentage = (MODEL_ENERGY_COST[LARGE_MODEL] - MODEL_ENERGY_COST[model]) / MODEL_ENERGY_COST[LARGE_MODEL] * 100
    includes_hints = input_data.includes_hints

    reasoning = f"The input prompt contains {token_num} tokens and references {number_of_files} file(s)."
    reasoning += f" The intent detected is {intent} which makes it classified as {complexity} complexity."
    reasoning += f" Considering the complexity, the suggested model is {model} which has an energy cost of {suggested_model_energy} units."
    reasoning += f" By choosing this model, compared to the baseline of {LARGE_MODEL} with an energy cost of {large_model_energy} units, the estimated savings is {savings} units; therefore, approximately {savings_percentage:.2f}% savings."

    if includes_hints:
        reasoning += f" The additional hints are considered in the assessment."

    return reasoning

def evaluating_swebench_difficulty(input_data: InputData) -> Optional[str]:
    if not input_data.swe_bench_difficulty:
        return "unknown"
    difficulty = input_data.swe_bench_difficulty.strip().lower()
    if difficulty == "<15 min fix":
        return "easy"
    if difficulty == "15 min - 1 hour":
        return "medium"
    if difficulty == "1-4 hours":
        return "hard"
        
    return None



def main(input_data: InputData) -> OutputData:
    intent = intent_detection(input_data)
    complexity = complexity_assessment(input_data)
    model = model_selection(input_data)
    model_name = MAPPING_TO_MODEL[model]
    estimated_saved_energy, savings_percentage = energy_savings_estimation(input_data, model)
    reasoning = reasoning_explanation(input_data, model)
    swe_bench_difficulty = evaluating_swebench_difficulty(input_data)
    
    return OutputData(intent=intent, complexity=complexity, model=model, model_name=model_name, estimated_saved_energy=estimated_saved_energy, reasoning=reasoning, savings_percentage=savings_percentage, swe_bench_difficulty=swe_bench_difficulty)


def checking_alignment(n: int) -> None:
    matches = 0
    total = 0

    for i in range(min(n, len(ds["test"]))):
        row = ds["test"][i]
        input_data = converter(row)
        output = main(input_data)

        if output.swe_bench_difficulty in {"easy", "medium", "hard"}:
            total += 1
            if output.complexity == output.swe_bench_difficulty:
                matches += 1

            else:
                print("\nMismatch")
                print("Problem:", input_data.problem_statement[:500])
                print("Predicted:", output.complexity)
                print("Benchmark:", output.swe_bench_difficulty)
                print("Intent:", output.intent)
                print("Tokens:", input_data.token_count)
                print("Includes hints:", input_data.includes_hints)
    
    print(f"Matches: {matches}/{total}")
    if total > 0:
        print(f"Alignment: {matches / total:.2%}")


if __name__ == "__main__":
    checking_alignment(500)