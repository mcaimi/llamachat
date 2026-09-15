#!/usr/bin/env python

from dataclasses import dataclass, field


@dataclass
class SkillSummary:
    name: str = ""
    description: str = ""
    directory: str = ""


@dataclass
class SkillDefinition:
    name: str = ""
    description: str = ""
    directory: str = ""
    instructions: str = ""
    allowed_tools: str = ""
    license: str = ""
    compatibility: str = ""
    metadata: dict = field(default_factory=dict)
