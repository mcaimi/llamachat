#!/usr/bin/env python

import os
import json
import re
import tempfile
from typing import Any, Iterable, Optional

try:
    import requests
    from streamlit import warning
    from .utils import build_header
    from .state import AgentMessage
except Exception as e:
    raise e


class Session(object):
    def __init__(self, session_state):
        self.streamlit_session = session_state
        self.session_state = self.streamlit_session

    def _ensure_history_dir(self) -> str:
        history_dir = getattr(self.streamlit_session, "history_dir", None)
        if not history_dir or not isinstance(history_dir, str):
            raise ValueError("history_dir is not configured.")
        os.makedirs(history_dir, exist_ok=True)
        return history_dir

    def _safe_filename(self, filename: str, required_ext: Optional[str] = None) -> str:
        if not filename or not isinstance(filename, str):
            raise ValueError("Filename is required.")

        trimmed = filename.strip()
        if not trimmed:
            raise ValueError("Filename is empty.")

        # block traversal / absolute paths on *nix + Windows, and NULs
        if "\x00" in trimmed:
            raise ValueError("Invalid filename.")
        if os.path.isabs(trimmed):
            raise ValueError("Absolute paths are not allowed.")
        if trimmed != os.path.basename(trimmed):
            raise ValueError("Path separators are not allowed in filename.")
        if ".." in trimmed:
            raise ValueError("Parent path segments are not allowed in filename.")

        # Restrict to a conservative character set; keep it readable.
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", trimmed)
        safe = safe.lstrip("._-")
        if not safe:
            raise ValueError("Filename is invalid.")

        if required_ext:
            ext = required_ext if required_ext.startswith(".") else f".{required_ext}"
            if not safe.lower().endswith(ext.lower()):
                safe = f"{safe}{ext}"

        return safe

    def _history_path(self, filename: str, required_ext: Optional[str] = None) -> str:
        history_dir = self._ensure_history_dir()
        safe_name = self._safe_filename(filename, required_ext=required_ext)
        return os.path.join(history_dir, safe_name)

    def save_chat_history(self, filename, chat_data) -> str:
        final_path = self._history_path(filename, required_ext=".json")

        json_document = []
        for item in chat_data:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            json_document.append({"role": role, "content": content})

        # Atomic write to avoid corrupting autosave on interruption.
        history_dir = os.path.dirname(final_path)
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=history_dir,
            prefix=".tmp_chat_",
            suffix=".json",
        )
        try:
            with tmp_file as f:
                json.dump(json_document, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file.name, final_path)
        finally:
            try:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
            except Exception:
                pass

        return final_path

    def export_chat_to_markdown(self, md_filename, chat_data):
        markdown_output = ""
        for msg in chat_data:
            role = msg.role.capitalize()
            content = msg.content
            markdown_output += f"### {role}\n\n{content}\n\n"
        final_path = self._history_path(md_filename, required_ext=".md")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        return final_path

    def load_chat_history(self, filename):
        # rebuild chat history
        chat_history = []
        path = self._history_path(filename, required_ext=".json")
        with open(path, "r", encoding="utf-8") as f:
            doc: Any = json.load(f)

        # Backward/forward compatible: accept either a list of messages or
        # a top-level object containing a `messages` array.
        if isinstance(doc, dict) and isinstance(doc.get("messages"), list):
            messages: Iterable[Any] = doc.get("messages", [])
        elif isinstance(doc, list):
            messages = doc
        else:
            raise ValueError("Unrecognized chat history format.")

        for item in messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role is None or content is None:
                continue
            chat_history.append(AgentMessage(_content=content, _role=role))

        # return rebuilt history
        return chat_history

    def list_saved_histories(self):
        history_dir = self._ensure_history_dir()
        files = [f for f in os.listdir(history_dir) if f.endswith(".json")]
        files.sort(
            key=lambda name: os.path.getmtime(os.path.join(history_dir, name)),
            reverse=True,
        )
        return files

    def models_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/v1/models"

    def chat_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/v1/chat/completions"

    def providers_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/v1/providers"

    # LIST METHODS
    def list_providers(
        self, provider_type: str = "vector_io", timeout: int = 10
    ) -> list:
        detected_providers = []
        if provider_type not in ["inference", "vector_io", "agents"]:
            return []

        try:
            resp = requests.get(
                self.providers_endpoint(),
                timeout=timeout,
                headers=build_header(self.session_state.api_key),
            )

            if resp.status_code == 200:
                detected_providers = [
                    m["provider_id"]
                    for m in resp.json().get("data", [])
                    if m["api"] == provider_type
                ]

            return detected_providers
        except Exception as e:
            warning("Could not fetch providers.")
            return None

    def list_models(self, model_type: str = "llm", timeout: int = 10) -> list:
        detected_models = []
        if model_type not in ["llm", "embedding"]:
            return []

        try:
            resp = requests.get(
                self.models_endpoint(),
                timeout=timeout,
                headers=build_header(self.session_state.api_key),
            )

            if resp.status_code == 200:
                models = [
                    m["id"]
                    for m in resp.json().get("data", [])
                    if m["custom_metadata"]["model_type"] == model_type
                ]
                if models:
                    detected_models = models
            return detected_models
        except Exception:
            warning("Could not fetch models. Using fallback.")
            return None

    def add_to_session_state(self, key, value) -> None:
        if key not in self.streamlit_session:
            setattr(self.streamlit_session, key, value)

    def remove_from_session_state(self, key) -> None:
        if key in self.streamlit_session:
            del self.streamlit_session[key]

    def clear_chat_session(self) -> None:
        self.session_state.agent_messages = []

    def update_system_prompt(self, new_prompt: str) -> None:
        self.session_state.system_prompt = new_prompt
