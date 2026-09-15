#!/usr/bin/env python

import os
from libs.agents.frontmatter import parse_frontmatter_only, parse_frontmatter
from libs.skills.definition import SkillSummary, SkillDefinition


def discover_skills(paths: list[str]) -> list[SkillSummary]:
    """Stage 1: Scan skill directories and read only frontmatter (~100 tokens per skill)."""
    skills = []
    for base_path in paths:
        expanded = os.path.expanduser(base_path)
        if not os.path.isdir(expanded):
            continue
        for entry in sorted(os.listdir(expanded)):
            skill_dir = os.path.join(expanded, entry)
            skill_file = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isdir(skill_dir) or not os.path.isfile(skill_file):
                continue
            try:
                fm = parse_frontmatter_only(skill_file)
                if fm.get("name") and fm.get("description"):
                    skills.append(
                        SkillSummary(
                            name=fm["name"],
                            description=fm["description"],
                            directory=skill_dir,
                        )
                    )
            except Exception:
                continue
    return skills


def activate_skill(skill_dir: str) -> SkillDefinition:
    """Stage 2: Read the full SKILL.md body for activation."""
    skill_file = os.path.join(skill_dir, "SKILL.md")
    fm, body = parse_frontmatter(skill_file)
    return SkillDefinition(
        name=fm.get("name", ""),
        description=fm.get("description", ""),
        directory=skill_dir,
        instructions=body,
        allowed_tools=fm.get("allowed-tools", ""),
        license=fm.get("license", ""),
        compatibility=fm.get("compatibility", ""),
        metadata=fm.get("metadata", {}),
    )


def load_skill_assets(skill_dir: str, filename: str) -> str:
    """Stage 3: Read a file from the skill's subdirectories on demand."""
    asset_path = os.path.join(skill_dir, filename)
    real_skill_dir = os.path.realpath(skill_dir)
    real_asset = os.path.realpath(asset_path)
    if not real_asset.startswith(real_skill_dir + os.sep):
        raise ValueError("Asset path escapes skill directory.")
    with open(asset_path, "r", encoding="utf-8") as f:
        return f.read()
