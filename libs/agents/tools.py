#!/usr/bin/env python

import json
import subprocess


SHELL_TOOL = {
    "type": "function",
    "name": "execute_command",
    "description": (
        "Execute a shell command on the local system. "
        "Use this to run scripts, inspect files, install packages, "
        "or perform system operations requested by the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "working_directory": {
                "type": "string",
                "description": "Working directory for command execution. Defaults to current directory.",
            },
        },
        "required": ["command"],
    },
}


def run_command(command: str, timeout: int = 30, cwd: str = None) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        output += f"\nExit code: {result.returncode}"
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {e}"


def extract_function_calls(response) -> list[dict]:
    calls = []
    for item in response.output:
        if item.type == "function_call":
            calls.append(
                {
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
            )
    return calls


def parse_call_arguments(arguments) -> dict:
    if isinstance(arguments, str):
        return json.loads(arguments)
    return arguments
