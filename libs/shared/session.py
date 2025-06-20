#!/usr/bin/env python

import os, json
try:
    import requests
    from streamlit import warning, session_state
    from .settings import Properties
except Exception as e:
    raise e

class Session(object):
    def __init__(self):
        self.streamlit_session = session_state
        self.session_state = self.streamlit_session

    def save_chat_history(self, filename, chat_data):
        with open(os.path.join(self.streamlit_session.history_dir, filename), "w") as f:
            json.dump(chat_data, f, indent=2)

    def load_chat_history(self, filename):
        with open(os.path.join(self.streamlit_session.history_dir, filename), "r") as f:
            return json.load(f)

    def list_saved_histories(self):
        return [f for f in os.listdir(self.streamlit_session.history_dir) if f.endswith(".json")]

    def models_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/tags"

    def chat_endpoint(self) -> str:
        return f"{self.streamlit_session.api_base_url}/chat"

    def list_ollama_models(self, timeout: int = 10) -> list:
        detected_models = []
        try:
            resp = requests.get(self.models_endpoint(), timeout=timeout)

            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if models:
                    detected_models = models
            return detected_models
        except Exception:
            warning("Could not fetch models. Using fallback.")
            return None

    def add_to_session_state(self, key, value) -> None:
        if key not in self.streamlit_session:
            setattr(self.streamlit_session, key, value)

    def update_system_prompt(self, new_prompt: str) -> None:
        self.session_state.system_prompt = new_prompt
        for i, msg in enumerate(self.session_state.messages):
            if msg['role'] == 'system':
                self.session_state.messages[i]['content'] = new_prompt
