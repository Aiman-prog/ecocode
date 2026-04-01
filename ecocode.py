#!/usr/bin/env python3
"""ecocode.py — Interactive CLI that routes prompts to the most energy-efficient LLM."""

import asyncio
import os
import sys
from rich import print
from rich.panel import Panel
from rich.text import Text

import litellm
from dotenv import load_dotenv

from logic import manage_input
from ecocode_executor import run as executor_run

LITELLM_MODEL_MAP = {
    "small":  "groq/llama-3.1-8b-instant",
    "medium": "groq/llama-3.3-70b-versatile",
    "large":  "groq/llama-3.3-70b-versatile",
}

def print_welcome() -> None:
    text = Text()
    text.append("EcoCode CLI\n", style="bold green")
    text.append("Energy-aware routing for coding tasks\n\n")

    text.append("Usage:\n", style="bold")
    text.append("  • Ask general questions\n")
    text.append("    ecocode> explain recursion\n\n")

    text.append("  • Fix code (requires full file path)\n")
    text.append("    ecocode> fix the bug in /full/path/to/file.py\n\n")

    text.append("Type 'exit' to quit.\n", style="dim")
    text.append("Type 'help' to display this message again.\n", style="dim")

    print(Panel(text, border_style="green"))


def print_dashboard(prompt: str, result: dict) -> None:
    text = Text()

    text.append("Task: ", style="bold")
    if len(prompt) > 60:
        text.append(f"{prompt[:60]}...\n")
    else:
        text.append(f"{prompt[:60]}\n")

    text.append("\nContext: ", style="bold")
    if result["context_files"]:
        text.append("File detected\n", style="green")
    else:
        text.append("none\n", style="red")

    text.append("\nComplexity: ", style="bold")
    complexity_color = {
        "easy": "green",
        "medium": "yellow",
        "hard": "red"
    }.get(result["complexity"].lower(), "white")
    text.append(f"{result['complexity']}\n", style=complexity_color)

    text.append("Model: ", style="bold")
    text.append(f"{LITELLM_MODEL_MAP[result['model_key']]}\n", style="cyan")

    text.append("\nEnergy: ", style="bold")
    text.append(f"{result['energy']:.6f} kWh\n")

    text.append("Savings: ", style="bold")
    text.append(f"{result['savings']:.2f}%\n", style="green")

    text.append("\nTokens: ", style="bold")
    text.append(f"{result['tokens']:.0f}\n")

    print(Panel(text, title="Eco Dashboard", border_style="blue"))



async def chat(prompt: str, model: str) -> None:
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)


def main() -> None:
    load_dotenv()

    print_welcome() 

    while True:
        try:
            prompt = input("\necocode> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if prompt.lower() == "help":
            print_welcome()
            continue

        result = manage_input(prompt)
        print_dashboard(prompt, result)

        model_key = result["model_key"]
        litellm_model = LITELLM_MODEL_MAP[model_key]

        if result["context_files"]:
            filepath = result["context_files"][0]
            os.environ["ECOCODE_MODEL"] = litellm_model
            try:
                asyncio.run(executor_run(prompt, filepath, verbose=False))
                print(f"File updated: {filepath}")
            except SystemExit:
                pass  # executor_run calls sys.exit on errors; already printed
        else:
            try:
                asyncio.run(chat(prompt, litellm_model))
            except litellm.exceptions.AuthenticationError:
                print("Error: LLM authentication failed — check your API key env vars.")
            except litellm.exceptions.APIConnectionError as exc:
                print(f"Error: LLM API connection error — {exc}")
            except litellm.exceptions.APIError as exc:
                print(f"Error: LLM API error — {exc}")


if __name__ == "__main__":
    main()