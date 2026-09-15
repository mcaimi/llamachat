#!/usr/bin/env python

from dataclasses import dataclass, field


@dataclass
class AgentDefinition:
    NONE_AGENT_NAME: str = "(default)"

    name: str = ""
    description: str = ""
    instructions: str = ""
    preferred_model: str = None
    mcp_servers: list = field(default_factory=list)
    rag: dict = field(default_factory=dict)
    skills: list = field(default_factory=list)
    model_parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    directory: str = ""

    def to_sampling_params(self) -> dict:
        if not self.model_parameters:
            return {}
        param_map = {
            "temperature": "temperature",
            "timeout": "timeout",
            "max_infer_iters": "max_infer_iters",
            "max_tool_calls": "max_tool_calls",
            "parallel_tool_calls": "parallel_tool_calls",
        }
        result = {}
        for src_key, dst_key in param_map.items():
            if src_key in self.model_parameters:
                result[dst_key] = self.model_parameters[src_key]
        return result

    def to_system_prompt(self) -> str:
        return self.instructions
