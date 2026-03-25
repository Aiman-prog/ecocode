import argparse
import os
import re

def manage_input(prompt):
    ##testing case
    if "--test" in prompt:
        return {
            "complexity": "Simple",
            "model": "Small",
            "energy": 0.0012,
            "baseline": 0.0058,
            "savings": 79,
            "tokens": 320
        }
    
    ##dummy logic to be replaced by task 2
    potential_paths = re.findall(r'(/[^\s,]+|[^\s,]+\.[a-zA-Z0-9]+)', prompt)
    files = [p.strip(',.?!') for p in potential_paths if os.path.exists(p.strip(',.?!'))]
    file_tokens = 0
    if files:
        with open(files[0], 'r') as f:
            file_tokens = len(f.read().split()) * 4
    else:
        files = "None"

    if len(prompt) < 50:
        complexity = "Simple"
        model = "Small"
    elif len(prompt) < 150:
        complexity = "Moderate"
        model = "Medium"
    else:
        complexity = "Complex"
        model = "Large"
    tokens = (len(prompt.split()) * 4) + file_tokens
    energy = tokens * 0.000005
    baseline = tokens * 0.00001
    savings = ((baseline-energy)/baseline) * 100
    return {
            "complexity": complexity,
            "model": model,
            "energy": energy,
            "baseline": baseline,
            "savings": savings,
            "tokens": tokens,
            "context_files": files
        }

def main():
    parser = argparse.ArgumentParser(description="EcoRoute CLI")
    parser.add_argument("prompt", type=str)
    args = parser.parse_args()

    result = manage_input(args.prompt)

    print("\n" + "="*40)
    print("    ECOCODE SUSTAINABILITY DASHBOARD")
    print("="*40)
    print(f"TASK DETECTED:    {args.prompt[:30]}...")
    if result['context_files']:
        print(f"CONTEXT AUDIT:    Found file(s): {result['context_files']}")
    else:
        print(f"CONTEXT AUDIT: NO FILE MENTIONED")
    print("-" * 40)
    print(f"COMPLEXITY:       {result['complexity']}")
    print(f"RECOMMENDED:      {result['model']}")
    print("-" * 40)
    print(f"EST. ENERGY:      {result['energy']:.6f}".rstrip('0').rstrip('.') + " kWh")
    print(f"SAVINGS:          {result['savings']}% ")
    print(f"EST. TOKENS:      {result['tokens']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()