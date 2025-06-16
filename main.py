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
from libs.shared.utils import *

# MAIN
if __name__ == "__main__":

    # load environment
    config_env: dict = dotenv_values(".env")

    # load app settings
    appSettings = Properties(dotenv_value_dict=config_env)

    # initialize streamlit session
    st.session_state.api_base_url = appSettings.api_base_url
    st.session_state.default_local_api = appSettings.default_local_api
    st.session_state.default_cloud_api = appSettings.default_cloud_api
    st.session_state.custom_endpoint = appSettings.custom_endpoint
    add_to_session_state(st.session_state, "messages", appSettings.messages)
    st.session_state.enable_rag = appSettings.enable_rag
    st.session_state.system_prompt = appSettings.system_prompt
    st.session_state.history_dir = appSettings.history_dir
    st.session_state.latest_history_filename = appSettings.latest_history_filename
    st.session_state.chromadb_host = appSettings.chromadb_host
    st.session_state.chromadb_collection = appSettings.chromadb_collection

    # build streamlit UI
    st.set_page_config(page_title="🧠 Ollama Chat Pro", layout="centered")
    st.html("assets/header.html")

    # get the configured api endpoint from session
    def get_chat_endpoint() -> str:
        return f"{st.session_state.api_base_url}/chat"

    # get the models api endpoint
    def get_models_endpoint() -> str:
        return f"{st.session_state.api_base_url}/tags"

    # Sidebar
    with st.sidebar:
        st.header("🛠 Settings")

        endpoint_choice = st.radio("🌐 Select API Endpoint", ["Local", "Cloud", "Custom"])

        if endpoint_choice == "Local":
            st.session_state.api_base_url = st.session_state.default_local_api
        elif endpoint_choice == "Cloud":
            st.session_state.api_base_url = st.session_state.default_cloud_api
        elif endpoint_choice == "Custom":
            st.session_state.custom_endpoint = st.text_input("🔧 Custom Endpoint", value=st.session_state.custom_endpoint)
            if st.session_state.custom_endpoint:
                st.session_state.api_base_url = st.session_state.custom_endpoint

        st.markdown(f"**🔌 Current Endpoint:** `{st.session_state.api_base_url}`")

        models = list_ollama_models(get_models_endpoint())
        if models:
            model_name = st.selectbox("🔁 Select Model", models)
        else:
            model_name = st.selectbox("🔁 Select Model", appSettings.available_models)

        st.markdown("---")
        st.subheader("📜 System Prompt")
        new_prompt = st.text_area("Update System Prompt", value=st.session_state.system_prompt, height=100)
        if st.button("🔄 Apply New Prompt"):
            st.session_state.system_prompt = new_prompt
            st.session_state.messages.insert(0, {"role": "system", "content": new_prompt})
            st.success("System prompt updated.")

        st.subheader("🧪 Model Parameters")
        temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.05)
        top_k = st.slider("🎯 Top-K", 0, 100, 40, 1)
        top_p = st.slider("📊 Top-P", 0.0, 1.0, 0.9, 0.05)
        repeat_penalty = st.slider("🔁 Repeat Penalty", 0.5, 2.0, 1.1, 0.1)

        st.markdown("---")
        save_name = st.text_input("💾 Filename to Save", value=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        if st.button("💾 Save History"):
            save_chat_history(st.session_state, save_name, st.session_state.messages)
            st.success(f"Saved to {save_name}")

        history_files = list_saved_histories(st.session_state)
        selected_file = st.selectbox("📂 Load History", ["-- Select --"] + history_files)
        if selected_file != "-- Select --" and st.button("📂 Load"):
            st.session_state.messages = load_chat_history(st.session_state, selected_file)
            st.success(f"Loaded {selected_file}")

        if os.path.exists(os.path.join(st.session_state.history_dir, st.session_state.latest_history_filename)):
            if st.button("🕓 Load Latest Chat"):
                st.session_state.messages = load_chat_history(st.session_state, st.session_state.latest_history_filename)
                st.success("Latest chat loaded!")

        if st.button("📤 Export to Markdown"):
            markdown_output = ""
            for msg in st.session_state.messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                markdown_output += f"### {role}\n\n{content}\n\n"

            md_filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(os.path.join(st.session_state.history_dir, md_filename), "w", encoding="utf-8") as f:
                f.write(markdown_output)
            st.success(f"Exported to {md_filename}")

        st.markdown("---")
        st.subheader("📚 RAG Settings")
        st.session_state.enable_rag = st.checkbox("🔎 Enable RAG", value=st.session_state.enable_rag)
        chroma_host = st.text_input("🌐 ChromaDB Host", value=st.session_state.chromadb_host)
        collection_name = st.text_input("📂 ChromaDB Collection", value=st.session_state.chromadb_collection)

    # Chat Interface
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

    prompt = st.chat_input("💬 Say something...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # RAG Phase
        if st.session_state.enable_rag:
            with st.spinner("🔎 Retrieving documents..."):
                retrieved_docs = retrieve_relevant_docs(prompt, st.session_state.chroma_host, st.session_state.chromadb_collection)
                if retrieved_docs:
                    context = "\n\n".join(retrieved_docs)
                    st.session_state.messages.append({
                           "role": "system",
                           "content": f"Use the following retrieved documents to answer the user's question:\n\n{context}"
                       })
                else:
                    st.warning("No relevant documents found.")

        # Assistant reply container
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            try:
                with requests.post(get_chat_endpoint(), json={"model": model_name, "messages": st.session_state.messages,
                                                                          "stream": True, "options": {"temperature": temperature,
                                                                                                      "top_k": top_k,
                                                                                                      "top_p": top_p,
                                                                                                      "repeat_penalty": repeat_penalty}}, stream=True, timeout=10) as resp:
                    if resp.status_code != 200:
                        st.error(f"API error: {resp.status_code}")
                    else:
                        for line in resp.iter_lines():
                            if line:
                                try:
                                    part = line.decode('utf-8').strip()
                                    if part.startswith("data: "):
                                        part = part[6:]
                                    data = json.loads(part)
                                    token = data.get("message", {}).get("content", "")
                                    full_response += token
                                    response_container.markdown(full_response, unsafe_allow_html=True)
                                except Exception as e:
                                    st.warning(f"Error parsing stream: {e}")
            except Exception as e:
                st.error(f"Request failed: {e}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            print(st.session_state.messages)

            save_chat_history(st.session_state, st.session_state.latest_history_filename, st.session_state.messages)

