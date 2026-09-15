#!/usr/bin/env python

from yaml import safe_load, YAMLError


def parse_frontmatter_only(filepath: str) -> dict:
    """Read only the YAML frontmatter from a Markdown file, without loading the body."""
    lines = []
    found_open = False
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "---":
                if not found_open:
                    found_open = True
                    continue
                else:
                    break
            if found_open:
                lines.append(line)
    if not lines:
        return {}
    try:
        return safe_load("\n".join(lines)) or {}
    except YAMLError:
        return {}


def parse_frontmatter(filepath: str) -> tuple[dict, str]:
    """Read the full file, return (frontmatter_dict, markdown_body)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        frontmatter = safe_load(parts[1]) or {}
    except YAMLError:
        frontmatter = {}

    body = parts[2].strip()
    return frontmatter, body
