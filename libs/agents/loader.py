#!/usr/bin/env python

import os
from libs.agents.frontmatter import parse_frontmatter_only, parse_frontmatter
from libs.agents.definition import AgentDefinition


def discover_agents(paths: list[str]) -> list[AgentDefinition]:
    """Scan agent directories and read only frontmatter from each AGENT.md."""
    agents = []
    for base_path in paths:
        expanded = os.path.expanduser(base_path)
        if not os.path.isdir(expanded):
            continue
        for entry in sorted(os.listdir(expanded)):
            agent_dir = os.path.join(expanded, entry)
            agent_file = os.path.join(agent_dir, "AGENT.md")
            if not os.path.isdir(agent_dir) or not os.path.isfile(agent_file):
                continue
            try:
                fm = parse_frontmatter_only(agent_file)
                agent_name = fm.get("name") or entry
                description = fm.get("description", "")
                if description:
                    agents.append(
                        AgentDefinition(
                            name=agent_name,
                            description=description,
                            directory=agent_dir,
                        )
                    )
            except Exception:
                continue
    return agents


_FLAT_PARAM_KEYS = {"temperature", "max_output_tokens", "max_infer_iters",
                    "max_tool_calls", "parallel_tool_calls", "timeout"}


def load_agent_definition(agent_dir: str) -> AgentDefinition:
    """Read the full AGENT.md and return a fully populated AgentDefinition."""
    agent_file = os.path.join(agent_dir, "AGENT.md")
    fm, body = parse_frontmatter(agent_file)
    dir_name = os.path.basename(agent_dir)

    model_parameters = fm.get("model_parameters", {})
    if not model_parameters:
        model_parameters = {k: fm[k] for k in _FLAT_PARAM_KEYS if k in fm}

    return AgentDefinition(
        name=fm.get("name") or dir_name,
        description=fm.get("description", ""),
        instructions=body,
        preferred_model=fm.get("preferred_model") or fm.get("model"),
        mcp_servers=fm.get("mcp_servers", []),
        rag=fm.get("rag", {}),
        skills=fm.get("skills", []),
        model_parameters=model_parameters,
        metadata=fm.get("metadata", {}),
        directory=agent_dir,
    )
