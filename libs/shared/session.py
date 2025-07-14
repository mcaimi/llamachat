#!/usr/bin/env python

import os, json
try:
    import requests
    from datetime import datetime
    from streamlit import warning, session_state
    from llama_stack_client.types import UserMessage, SystemMessage, CompletionMessage
    from .utils import build_header
except Exception as e:
    raise e

class Session(object):
    def __init__(self):
        self.streamlit_session = session_state
        self.session_state = self.streamlit_session

    def save_chat_history(self, filename, chat_data):
        with open(os.path.join(self.streamlit_session.history_dir, filename), "w") as f:
            json_document = []
            for item in chat_data:
                if type(item) == CompletionMessage:
                    json_document.append({"role": item.role, "content": item.content, "stop_reason": item.stop_reason})
                else:
                    json_document.append({"role": item.role, "content": item.content})

            # dump data to json    
            json.dump(json_document, f, indent=2)

    def export_chat_to_markdown(self, md_filename, chat_data):
        markdown_output = ""
        for msg in chat_data:
            role = msg.role.capitalize()
            content = msg.content
            markdown_output += f"### {role}\n\n{content}\n\n"
            # export to markdown
            with open(os.path.join(self.session_state.history_dir, md_filename), "w", encoding="utf-8") as f:
                f.write(markdown_output)

    def load_chat_history(self, filename):
        # rebuild chat history
        chat_history = []
        with open(os.path.join(self.streamlit_session.history_dir, filename), "r") as f:
            json_document = json.load(f)
            # iterate over messages          
            for item in json_document:
                if type(item) == dict and "content" in item:
                    match item["role"]:
                        case "system":
                            chat_history.insert(0, SystemMessage(content=item["content"], role="system"))
                        case "user":
                            chat_history.append(UserMessage(content=item["content"], role="user"))
                        case "assistant":
                            chat_history.append(CompletionMessage(content=item["content"], role="assistant", stop_reason=item["stop_reason"]))

        # return rebuilt history
        return chat_history    

    def list_saved_histories(self):
        return [f for f in os.listdir(self.streamlit_session.history_dir) if f.endswith(".json")]

    def models_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/v1/models"

    def chat_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/v1/chat/completions"

    def shields_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/v1/shields"

    def providers_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/v1/providers"

    def list_providers(self, provider_type: str = "vector_io", timeout: int = 10) -> list:
        detected_providers = []
        if provider_type not in ["inference", "vector_io", "agents"]:
            return []

        try:
            resp = requests.get(self.providers_endpoint(), timeout=timeout, headers=build_header(self.session_state.api_key))

            if resp.status_code == 200:
                detected_providers = [m['provider_id'] for m in resp.json().get('data', []) if m['api'] == provider_type]

            return detected_providers
        except Exception as e:
            warning("Could not fetch providers.")
            return None            

    def list_models(self, model_type: str = "llm", timeout: int = 10) -> list:
        detected_models = []
        if model_type not in ["llm", "embedding", "shield"]:
            return []

        try:
            if model_type == "shield":
                resp = requests.get(self.shields_endpoint(), timeout=timeout, headers=build_header(self.session_state.api_key))
            else:
                resp = requests.get(self.models_endpoint(), timeout=timeout, headers=build_header(self.session_state.api_key))

            if resp.status_code == 200:
                if model_type == "shield":
                    models = [m['identifier'] for m in resp.json().get('data', []) if m['type'] == model_type]
                else:
                    models = [m['identifier'] for m in resp.json().get('data', []) if m['model_type'] == model_type]
                if models:
                    detected_models = models
            return detected_models
        except Exception:
            warning("Could not fetch models. Using fallback.")
            return None

    def add_to_session_state(self, key, value) -> None:
        if key not in self.streamlit_session:
            setattr(self.streamlit_session, key, value)

    def clear_chat_session(self) -> None:
        self.session_state.messages = [SystemMessage(content=self.session_state.system_prompt, role="system")]

    def update_system_prompt(self, new_prompt: str) -> None:
        self.session_state.system_prompt = new_prompt
        for i, msg in enumerate(self.session_state.messages):
            if type(msg) == SystemMessage:
                self.session_state.messages[i] = SystemMessage(content=new_prompt, role="system")
