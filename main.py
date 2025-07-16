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

# MAIN
if __name__ == "__main__":
    # define app pages
    ai_chat_page = st.Page("pages/ai_chat.py", title="AI Chat with RAG", icon=":material/chat:")
    agentic_page = st.Page("pages/agentic_chat.py", title="Agentic AI Chat", icon=":material/chat:")
    embeddings_page = st.Page("pages/embeddings.py", title="Manage Embeddings", icon=":material/search:")
    settings_page = st.Page("pages/settings.py", title="Application Settings", icon=":material/settings:")

    # setup application main page
    pg = st.navigation([ai_chat_page, agentic_page, embeddings_page, settings_page])
    st.set_page_config(page_title="Red Hat Opensource AI", page_icon=":material/edit:")

    # run app
    pg.run()
