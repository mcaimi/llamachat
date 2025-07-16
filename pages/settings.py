#!/usr/bin/env python

try:
    import streamlit as st
    import mimetypes as mt
    from llama_stack_client import RAGDocument, LlamaStackClient
    from dotenv import dotenv_values
    from libs.shared.settings import Properties
    from libs.shared.session import Session
    import pdfplumber
except ImportError as e:
    print(f"Caught Exception: {e}")

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# load session
stSession = Session(st.session_state)

# build streamlit UI
st.set_page_config(page_title="🧠 Application Settings", initial_sidebar_state="collapsed", layout="wide")
st.html("assets/settings.html")

# setting tabs
model_tab, shields_tab, sysprompt_tab, model_params_tab = st.tabs(["Model Selection", "Shields", "System Prompt", "Model Parameters"])

with model_tab:
    models = stSession.list_models(model_type="llm")
    if models:
        stSession.session_state.model_name = st.selectbox("🧠 Select Model", models)
    else:
        stSession.session_state.model_name = st.selectbox("🧠 Select Model", stSession.session_state.fallback_models)

    embedding_models = stSession.list_models(model_type="embedding")
    if embedding_models:
        stSession.session_state.embedding_model_name = st.selectbox("🔁 Select Embedding Model", embedding_models)
    else:
        stSession.session_state.embedding_model_name = st.selectbox("🔁 Select Embedding Model", stSession.session_state.fallback_models)

    api_key = st.text_input("🔐 API Key", value=stSession.session_state.api_key)
    if not api_key:
        st.warning("No API key provided")
    else:
        stSession.session_state.api_key = api_key

with shields_tab:
    stSession.session_state.enable_shields = st.checkbox("🔎 Enable Shields", value=stSession.session_state.enable_shields)
    shield_models = stSession.list_models(model_type="shield")
    if shield_models:
        stSession.session_state.shield_model_name = st.selectbox("🔁 Select Shield Model", shield_models)
    else:
        stSession.session_state.shield_model_name = st.selectbox("🔁 Select Shield Model", stSession.session_state.fallback_models)

with sysprompt_tab:
    new_prompt = st.text_area("Update System Prompt", value=stSession.session_state.system_prompt, height=150)
    if st.button("🔄 Apply New Prompt"):
        stSession.update_system_prompt(new_prompt=new_prompt)
        st.success("System prompt updated.")

with model_params_tab:
    stSession.session_state.temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.7, 0.05)
    stSession.session_state.top_p = st.slider("📊 Top-P", 0.0, 1.0, 0.9, 0.05)
    stSession.session_state.n_comp = st.text_input("Num Completions", value=appSettings.config_parameters.llm.n_comp)
    stSession.session_state.max_tokens = st.text_input("🔁 Tokens", value=appSettings.config_parameters.llm.max_tokens)
    stSession.session_state.presence_penalty = st.slider("🔁 Presence Penalty", -2.0, 2.0, 1.1, 0.1)
    stSession.session_state.repeat_penalty = st.slider("🔁 Repeat Penalty", -2.0, 2.0, 1.1, 0.1)
