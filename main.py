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

# MAIN
if __name__ == "__main__":

    # load environment
    config_env: dict = dotenv_values(".env")

    # load app settings
    appSettings = Properties(dotenv_value_dict=config_env)

    # initialize streamlit session
    stSession = Session()
    stSession.setup_session_state(appSettings)
    stSession.add_to_session_state("api_key", None)
    
    # build streamlit UI
    st.set_page_config(page_title="🧠 RedHat AI Assistant", layout="wide")
    st.html("assets/header.html")

    # get the configured api endpoint from session
    def get_chat_endpoint() -> str:
        return f"{stSession.session_state.api_base_url}/chat"

    # get the models api endpoint
    def get_models_endpoint() -> str:
        return f"{stSession.session_state.api_base_url}/tags"

    # Sidebar
    with st.sidebar:
        st.header("🛠 LLM Settings")

        st.markdown(f"**🔌 Current Endpoint:** `{stSession.session_state.api_base_url}`")

        with st.expander("🧠 Model Selection"):
            endpoint_choice = st.radio("🌐 Select API Endpoint", ["Local", "Cloud", "Custom"])

            if endpoint_choice == "Local":
                stSession.session_state.api_base_url = stSession.session_state.default_local_api
            elif endpoint_choice == "Cloud":
                stSession.session_state.api_base_url = stSession.session_state.default_cloud_api
            elif endpoint_choice == "Custom":
                stSession.session_state.custom_endpoint = st.text_input("🔧 Custom Endpoint", value=stSession.session_state.custom_endpoint)
                if stSession.session_state.custom_endpoint:
                    stSession.session_state.api_base_url = stSession.session_state.custom_endpoint

            models = stSession.list_ollama_models()
            if models:
                model_name = st.selectbox("🔁 Select Model", models)
            else:
                model_name = st.selectbox("🔁 Select Model", appSettings.available_models)

            api_key = st.text_input("🔐 API Key", value=stSession.session_state.api_key)
            if not api_key:
                st.warning("No API key provided")
            else:
                stSession.session_state.api_key = api_key

        st.markdown("---")
        with st.expander("📜 System Prompt"):
            new_prompt = st.text_area("Update System Prompt", value=stSession.session_state.system_prompt, height=100)
            if st.button("🔄 Apply New Prompt"):
                stSession.session_state.system_prompt = new_prompt
                stSession.session_state.messages.insert(0, {"role": "system", "content": new_prompt})
                st.success("System prompt updated.")

        with st.expander("🧪 Model Parameters"):
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

        # RAG Phase
        if stSession.session_state.enable_rag:
            with st.spinner("🔎 Retrieving documents..."):
                retrieved_docs = retrieve_relevant_docs(prompt, stSession.session_state.chroma_host, stSession.session_state.chromadb_collection)
                if retrieved_docs:
                    context = "\n\n".join(retrieved_docs)
                    stSession.session_state.messages.append({
                           "role": "system",
                           "content": f"Use the following retrieved documents to answer the user's question:\n\n{context}"
                       })
                else:
                    st.warning("No relevant documents found.")

        # Assistant reply container
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            # prepare json payload.
            chat_payload: dict = {"model": model_name, "messages": stSession.session_state.messages,
                                  "temperature": temperature,
                                  "top_p": top_p,
                                  "stream": True,
                                  "n": n_comp,
                                  "max_tokens": max_tokens,
                                  "presence_penalty": presence_penalty,
                                  "repeat_penalty": repeat_penalty}
            
            # execute inference on chat endpoint
            try:
                with requests.post(get_chat_endpoint(), headers=build_header(stSession.session_state.api_key), json=chat_payload, stream=True, timeout=10) as resp:
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

            # append full response to chat history
            stSession.session_state.messages.append({"role": "assistant", "content": full_response})

            # save latest messages in the last_chat json file on disk
            stSession.save_chat_history(stSession.session_state.latest_history_filename, stSession.session_state.messages)

