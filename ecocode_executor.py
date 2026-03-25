#!/usr/bin/env python3
"""ecocode_executor.py — Read-Fix-Write loop via MCP + LiteLLM."""

import argparse
import asyncio
import logging
import os
import re
import sys

import litellm
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_MODEL = "deepseek/deepseek-chat"

SYSTEM_PROMPT = (
    "You are a code transformation assistant. "
    "Return ONLY the complete, updated source code. "
    "Do NOT wrap it in markdown code fences (no backticks). "
    "Do NOT include any commentary, explanation, or prose. "
    "Your entire response must be raw source code only."
)

# Strips ```[lang]\n...\n``` wrappers when the entire response is one fence block.
_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)\n```\s*$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a file, apply an LLM transformation, write it back — via MCP."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="The transformation instruction to send to the LLM.",
    )
    parser.add_argument(
        "--filepath",
        required=True,
        help="Absolute or relative path to the file to transform.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def clean_llm_output(raw: str) -> str:
    """Strip accidental markdown fences and surrounding whitespace."""
    stripped = raw.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped


async def run(prompt: str, filepath: str, verbose: bool) -> None:
    log = logging.getLogger(__name__)

    filepath = os.path.abspath(filepath)
    allowed_dir = os.path.dirname(filepath)
    log.debug("Resolved filepath: %s", filepath)
    log.debug("MCP allowed_dir: %s", allowed_dir)

    model = os.environ.get("ECOCODE_MODEL", DEFAULT_MODEL)
    log.debug("Using model: %s", model)

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", allowed_dir],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                log.info("MCP session initialized.")

                # STEP 1: READ
                log.info("Reading file via MCP: %s", filepath)
                try:
                    read_result = await session.call_tool(
                        "read_file", {"path": filepath}
                    )
                except Exception as exc:
                    log.error("MCP read_file failed: %s", exc)
                    sys.exit(1)

                if not read_result.content:
                    log.error("read_file returned empty content for: %s", filepath)
                    sys.exit(1)

                file_content = read_result.content[0].text
                log.debug("Read %d chars from file.", len(file_content))

                # STEP 2: LLM CALL
                user_message = (
                    f"User instruction: {prompt}\n\n"
                    f"File content:\n{file_content}"
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]

                log.info("Sending request to LLM (%s)...", model)
                try:
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                    )
                except litellm.exceptions.AuthenticationError as exc:
                    log.error(
                        "LLM authentication failed — check your API key env vars: %s", exc
                    )
                    sys.exit(1)
                except litellm.exceptions.APIConnectionError as exc:
                    log.error("LLM API connection error: %s", exc)
                    sys.exit(1)
                except litellm.exceptions.APIError as exc:
                    log.error("LLM API error: %s", exc)
                    sys.exit(1)

                raw_output = response.choices[0].message.content
                log.debug("Raw LLM output (%d chars).", len(raw_output))

                # STEP 3: CLEAN
                cleaned_code = clean_llm_output(raw_output)
                log.debug("Cleaned output (%d chars).", len(cleaned_code))

                # STEP 4: WRITE
                log.info("Writing updated file via MCP: %s", filepath)
                try:
                    await session.call_tool(
                        "write_file",
                        {"path": filepath, "content": cleaned_code},
                    )
                except Exception as exc:
                    log.error("MCP write_file failed: %s", exc)
                    sys.exit(1)

                log.info("Success — file updated: %s", filepath)

    except OSError as exc:
        log.error("Failed to launch MCP server (is npx installed?): %s", exc)
        sys.exit(1)


def main() -> None:
    load_dotenv()

    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run(args.prompt, args.filepath, args.verbose))


if __name__ == "__main__":
    main()
