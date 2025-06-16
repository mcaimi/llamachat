#!/usr/bin/env python

import os, json
try:
    import requests
    from streamlit import warning
except Exception as e:
    raise e

def save_chat_history(session, filename, chat_data):
    with open(os.path.join(session.history_dir, filename), "w") as f:
        json.dump(chat_data, f, indent=2)

def load_chat_history(session, filename):
    with open(os.path.join(session.history_dir, filename), "r") as f:
        return json.load(f)

def list_saved_histories(session):
    return [f for f in os.listdir(session.history_dir) if f.endswith(".json")]

def list_ollama_models(endpoint: str, timeout: int = 10) -> list:
    detected_models = []
    try:
        resp = requests.get(endpoint, timeout=timeout)
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json().get('models', [])]
            if models:
                detected_models = models
        return detected_models
    except Exception:
        warning("Could not fetch models. Using fallback.")
        return None

def add_to_session_state(session_state, key, value):
    if key not in session_state:
        setattr(session_state, key, value)