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
from libs.shared.session import Session
from libs.shared.utils import build_header

# llama stack
from llama_stack_client import LlamaStackClient

# MAIN
if __name__ == "__main__":

    # load environment
    config_env: dict = dotenv_values(".env")

    # load app settings
    config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
    appSettings = Properties(config_file=config_filename)

    # initialize streamlit session
    stSession = Session()
    # setup default values from config file
    stSession.add_to_session_state("api_base_url", appSettings.config_parameters.openai.default_local_api)
    stSession.add_to_session_state("fallback_models", ["granite3.3:2b"])
    stSession.add_to_session_state("custom_endpoint", "")
    stSession.add_to_session_state("system_prompt", appSettings.config_parameters.llm.system_prompt)
    stSession.add_to_session_state("history_dir", appSettings.config_parameters.openai.history_dir)
    stSession.add_to_session_state("latest_history_filename", appSettings.config_parameters.openai.latest_history_filename)
    stSession.add_to_session_state("api_key", appSettings.config_parameters.openai.api_key)
    stSession.add_to_session_state("enable_rag", appSettings.config_parameters.features.enable_rag)
    stSession.add_to_session_state("chromadb_host", appSettings.config_parameters.chromadb.host)
    stSession.add_to_session_state("chromadb_port", appSettings.config_parameters.chromadb.port)
    stSession.add_to_session_state("chromadb_collection", appSettings.config_parameters.chromadb.collection)
    stSession.add_to_session_state("messages", [{"role":"system", "content": stSession.session_state.system_prompt}])

    # build streamlit UI
    st.set_page_config(page_title="🧠 RedHat AI Assistant", initial_sidebar_state="collapsed", layout="wide")
    st.html("assets/header.html")

    # Sidebar
    with st.sidebar:
        st.header("🛠 LLM Settings")

        if st.button("Clear Current Chat"):
            stSession.clear_chat_session()

        st.markdown(f"**🔌 Current Endpoint:** `{stSession.session_state.api_base_url}`")

        with st.expander("🧠 Model Selection"):
            endpoint_choice = st.radio("🌐 Select API Endpoint", ["Local", "Cloud", "Custom"])

            if endpoint_choice == "Local":
                stSession.session_state.api_base_url = appSettings.config_parameters.openai.default_local_api
            elif endpoint_choice == "Cloud":
                stSession.session_state.api_base_url = appSettings.config_parameters.openai.default_cloud_api
            elif endpoint_choice == "Custom":
                stSession.session_state.custom_endpoint = st.text_input("🔧 Custom Endpoint", value=stSession.session_state.custom_endpoint)
                if stSession.session_state.custom_endpoint:
                    stSession.session_state.api_base_url = stSession.session_state.custom_endpoint

            models = stSession.list_models(model_type="llm")
            if models:
                model_name = st.selectbox("🧠 Select Model", models)
            else:
                model_name = st.selectbox("🧠 Select Model", stSession.session_state.fallback_models)

            embedding_models = stSession.list_models(model_type="embedding")
            if embedding_models:
                embedding_name = st.selectbox("🔁 Select Embedding Model", embedding_models)
            else:
                embedding_name = st.selectbox("🔁 Select Embedding Model", stSession.session_state.fallback_models)

            api_key = st.text_input("🔐 API Key", value=stSession.session_state.api_key)
            if not api_key:
                st.warning("No API key provided")
            else:
                stSession.session_state.api_key = api_key


        st.markdown("---")
        with st.expander("📜 AI Safety Shields"):
            shield_models = stSession.list_models(model_type="shield")
            if shield_models:
                shield_name = st.selectbox("🔁 Select Shield Model", shield_models)
            else:
                shield_name = st.selectbox("🔁 Select Shield Model", stSession.session_state.fallback_models)

        st.markdown("---")
        with st.expander("📜 System Prompt"):
            new_prompt = st.text_area("Update System Prompt", value=stSession.session_state.system_prompt, height=100)
            if st.button("🔄 Apply New Prompt"):
                stSession.update_system_prompt(new_prompt=new_prompt)
                st.success("System prompt updated.")

        st.markdown("---")
        with st.expander("🔧 Model Parameters"):
            temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.7, 0.05)
            top_p = st.slider("📊 Top-P", 0.0, 1.0, 0.9, 0.05)
            n_comp = st.text_input("Num Completions", value=1)
            max_tokens = st.text_input("🔁 Tokens", value=1024)
            presence_penalty = st.slider("🔁 Presence Penalty", -2.0, 2.0, 1.1, 0.1)
            repeat_penalty = st.slider("🔁 Repeat Penalty", -2.0, 2.0, 1.1, 0.1)

        st.markdown("---")
        with st.expander("💾 Filesystem IO"):
            save_name = st.text_input("💾 Filename to Save", value=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            if st.button("💾 Save History"):
                stSession.save_chat_history(save_name, stSession.session_state.messages)
                st.success(f"Saved to {save_name}")

            history_files = stSession.list_saved_histories()
            selected_file = st.selectbox("📂 Load History", ["-- Select --"] + history_files)
            if selected_file != "-- Select --" and st.button("📂 Load"):
                stSession.session_state.messages = stSession.load_chat_history(selected_file)
                st.success(f"Loaded {selected_file}")

            if os.path.exists(os.path.join(stSession.session_state.history_dir, stSession.session_state.latest_history_filename)):
                if st.button("🕓 Load Latest Chat"):
                    stSession.session_state.messages = stSession.load_chat_history(stSession.session_state.latest_history_filename)
                    st.success("Latest chat loaded!")

            if st.button("📤 Export to Markdown"):
                markdown_output = ""
                for msg in stSession.session_state.messages:
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    markdown_output += f"### {role}\n\n{content}\n\n"

                md_filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(os.path.join(stSession.session_state.history_dir, md_filename), "w", encoding="utf-8") as f:
                    f.write(markdown_output)
                st.success(f"Exported to {md_filename}")

        st.markdown("---")
        with st.expander("📚 RAG Settings"):
            stSession.session_state.enable_rag = st.checkbox("🔎 Enable RAG", value=stSession.session_state.enable_rag)
            chroma_host = st.text_input("🌐 ChromaDB Host", value=stSession.session_state.chromadb_host)
            collection_name = st.text_input("📂 ChromaDB Collection", value=stSession.session_state.chromadb_collection)

    # Chat Interface
    for msg in stSession.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

    prompt = st.chat_input("💬 Say something...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        stSession.session_state.messages.append({"role": "user", "content": prompt})

        # Assistant reply container
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            # execute inference on chat endpoint
            try:
                chatClient = LlamaStackClient(base_url=stSession.session_state.api_base_url)

                # if enabled, run safety shield
                shield_output = chatClient.safety.run_shield(
                    messages=[{"role": "user", "content": prompt}],
                    shield_id=shield_name,
                    params={}
                )

                # detect unappropriate prompt
                if shield_output.violation:
                    response_container.markdown("**Input Shielding**: {}".format(shield_output))
                else:
                    # perform inference
                    for line in chatClient.chat.completions.create(messages=stSession.session_state.messages, model=model_name, stream=True):
                        for event in line.choices:
                            token = event.delta.content
                            full_response += token
                        response_container.markdown(full_response, unsafe_allow_html=True)
                    
                    # append full response to chat history
                    stSession.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Request failed: {e}")

            # save latest messages in the last_chat json file on disk
            stSession.save_chat_history(stSession.session_state.latest_history_filename, stSession.session_state.messages)