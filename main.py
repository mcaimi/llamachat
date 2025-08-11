#!/usr/bin/env python
#
# LLM FRONTEND APP
# Streamlit version
#

import os
from datetime import datetime
import json

try:
    import requests
    import streamlit as st
    from dotenv import dotenv_values
except Exception as e:
    print(f"Caught fatal exception: {e}")

# local imports
from libs.shared.settings import Properties

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# MAIN
if __name__ == "__main__":
    # define app pages
    if appSettings.config_parameters.interface.agentic:
        agentic_page = st.Page("pages/agentic_chat.py", title="Agentic AI Chat", icon=":material/chat:")
        embeddings_page = st.Page("pages/embeddings.py", title="Manage Embeddings", icon=":material/search:")
        audio_page = st.Page("pages/whisper_audio.py", title="Audio to Text", icon=":material/speaker:")
        enabled_sections = [agentic_page, embeddings_page, audio_page]
    else:
        ai_chat_page = st.Page("pages/ai_chat.py", title="AI Chat with RAG", icon=":material/chat:")
        embeddings_page = st.Page("pages/embeddings.py", title="Manage Embeddings", icon=":material/search:")
        settings_page = st.Page("pages/settings.py", title="Application Settings", icon=":material/settings:")
        enabled_sections = [ai_chat_page, embeddings_page, settings_page]

    # setup application main page
    st.logo("assets/redhat.bmp")
    pg = st.navigation(enabled_sections)
    st.set_page_config(page_title="Red Hat Opensource AI", page_icon=":material/edit:")

    # run app
    pg.run()
