#!/usr/bin/env python
#
# Persistent State Trackers
#

from dataclasses import dataclass

@dataclass
class AgentMessage:
    _role: str = None
    _content: str = None

    @property
    def role(self) -> str:
        return self._role

    @role.setter
    def role(self, text: str) -> None:
        self._role = text

    @property
    def content(self) -> str:
        return self._content

    @content.setter
    def content(self, text: str) -> None:
        self._content = text