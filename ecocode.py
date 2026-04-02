#!/usr/bin/env python3
"""ecocode.py — Interactive CLI that routes prompts to the most energy-efficient LLM."""

import asyncio
import os
import sys

import litellm
from dotenv import load_dotenv

from logic import manage_input
from ecocode_executor import run as executor_run

LITELLM_MODEL_MAP = {
    "small":  "gemini/gemini-2.5-flash",
    "medium": "gemini/gemini-2.5-flash",
    "large":  "gemini/gemini-2.5-flash",
}


def print_dashboard(prompt: str, result: dict) -> None:
    print("\n" + "=" * 40)
    print("    ECOCODE SUSTAINABILITY DASHBOARD")
    print("=" * 40)
    print(f"TASK DETECTED:    {prompt[:30]}...")
    if result["context_files"]:
        print(f"CONTEXT AUDIT:    Found file(s): {result['context_files']}")
    else:
        print(f"CONTEXT AUDIT:    NO FILE FOUND")
    print("-" * 40)
    print(f"COMPLEXITY:       {result['complexity']}")
    print(f"RECOMMENDED:      {result['model']}")
    print(f"MODEL:            {LITELLM_MODEL_MAP[result['model_key']]}")
    print("-" * 40)
    print(f"EST. ENERGY:      {result['energy']:.6f}".rstrip("0").rstrip(".") + " kWh")
    print(f"SAVINGS:          {result['savings']:.2f}%")
    print(f"EST. TOKENS:      {result['tokens']:.0f}")
    print(f"REASONING:        {result['reasoning']}")
    print("=" * 40 + "\n")


async def chat(prompt: str, model: str) -> None:
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)


def main() -> None:
    load_dotenv()

    print("EcoCode — type your prompt, or 'exit' to quit.")

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