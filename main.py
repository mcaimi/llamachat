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
from llama_stack_client.types import UserMessage, SystemMessage, CompletionMessage

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
    stSession.add_to_session_state("shields_fallback_models", ["meta-llama/Llama-Guard-3-1b"])
    stSession.add_to_session_state("system_prompt", appSettings.config_parameters.llm.system_prompt)
    stSession.add_to_session_state("history_dir", appSettings.config_parameters.openai.history_dir)
    stSession.add_to_session_state("latest_history_filename", appSettings.config_parameters.openai.latest_history_filename)
    stSession.add_to_session_state("api_key", appSettings.config_parameters.openai.api_key)
    stSession.add_to_session_state("enable_rag", appSettings.config_parameters.features.enable_rag)
    stSession.add_to_session_state("enable_shields", appSettings.config_parameters.features.enable_shields)
    stSession.add_to_session_state("messages", [SystemMessage(role="system", content=stSession.session_state.system_prompt)])

    # session config
    stSession.add_to_session_state("model_name", appSettings.config_parameters.openai.model)
    stSession.add_to_session_state("embedding_model_name", appSettings.config_parameters.openai.embedding_model)
    stSession.add_to_session_state("shield_model_name", appSettings.config_parameters.openai.shield_model)
    stSession.add_to_session_state("temperature", appSettings.config_parameters.llm.temperature)
    stSession.add_to_session_state("top_p", appSettings.config_parameters.llm.top_p)
    stSession.add_to_session_state("n_comp", appSettings.config_parameters.llm.n_comp)
    stSession.add_to_session_state("max_tokens", appSettings.config_parameters.llm.max_tokens)
    stSession.add_to_session_state("presence_penalty", appSettings.config_parameters.llm.presence_penalty)
    stSession.add_to_session_state("repeat_penalty", appSettings.config_parameters.llm.repeat_penalty)
    stSession.add_to_session_state("vector_provider", appSettings.config_parameters.vectorstore.provider)
    stSession.add_to_session_state("collection_name", appSettings.config_parameters.vectorstore.collection)
    stSession.add_to_session_state("vectorstore_collection", appSettings.config_parameters.vectorstore.collection)

    # build streamlit UI
    st.set_page_config(page_title="🧠 RedHat AI Assistant", initial_sidebar_state="collapsed", layout="wide")
    st.html("assets/header.html")

    # Sidebar
    with st.sidebar:
        st.header("🛠 LLM Control Panel")

        if st.button("Clear Current Chat"):
            stSession.clear_chat_session()

        st.markdown(f"**🔌 Current Endpoint:** `{stSession.session_state.api_base_url}`")
        st.markdown(f"**🔌 Current Model:** `{stSession.session_state.model_name}`")
        st.markdown(f"**🔌 Current Embedding Model:** `{stSession.session_state.embedding_model_name}`")
        if stSession.session_state.enable_shields:
            st.markdown(f"**🔌 Shields Model:** `{stSession.session_state.shield_model_name}`")

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
                md_filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                stSession.export_chat_to_markdown(md_filename, stSession.session_state.messages)
                st.success(f"Exported to {md_filename}")

        with st.expander("🧠 Application Settings"):
            with st.expander("🧠 Model Selection"):
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

            st.markdown("---")
            with st.expander("📜 AI Safety Shields"):
                stSession.session_state.enable_shields = st.checkbox("🔎 Enable Shields", value=stSession.session_state.enable_shields)
                shield_models = stSession.list_models(model_type="shield")
                if shield_models:
                    stSession.session_state.shield_model_name = st.selectbox("🔁 Select Shield Model", shield_models)
                else:
                    stSession.session_state.shield_model_name = st.selectbox("🔁 Select Shield Model", stSession.session_state.fallback_models)

            st.markdown("---")
            with st.expander("📜 System Prompt"):
                new_prompt = st.text_area("Update System Prompt", value=stSession.session_state.system_prompt, height=150)
                if st.button("🔄 Apply New Prompt"):
                    stSession.update_system_prompt(new_prompt=new_prompt)
                    st.success("System prompt updated.")

            st.markdown("---")
            with st.expander("🔧 Model Parameters"):
                stSession.session_state.temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.7, 0.05)
                stSession.session_state.top_p = st.slider("📊 Top-P", 0.0, 1.0, 0.9, 0.05)
                stSession.session_state.n_comp = st.text_input("Num Completions", value=appSettings.config_parameters.llm.n_comp)
                stSession.session_state.max_tokens = st.text_input("🔁 Tokens", value=appSettings.config_parameters.llm.max_tokens)
                stSession.session_state.presence_penalty = st.slider("🔁 Presence Penalty", -2.0, 2.0, 1.1, 0.1)
                stSession.session_state.repeat_penalty = st.slider("🔁 Repeat Penalty", -2.0, 2.0, 1.1, 0.1)

        st.markdown("---")
        with st.expander("🔍 Agents Settings"):
            with st.expander("📚 Vector IO Providers"):
                providers = stSession.list_providers(provider_type="vector_io")
                if providers:
                    stSession.session_state.provider_name = st.selectbox("🧠 Select Provider", providers)
                else:
                    stSession.session_state.provider_name = st.selectbox("🧠 Select Provider", stSession.session_state.vector_provider)

            with st.expander("📚 RAG"):
                stSession.session_state.enable_rag = st.checkbox("🔎 Enable RAG", value=stSession.session_state.enable_rag)
                stSession.session_state.collection_name = st.selectbox("📂 ChromaDB Collection", [stSession.session_state.vectorstore_collection])

    # Chat Interface
    for msg in stSession.session_state.messages:
        if msg.role != "system":
            with st.chat_message(msg.role):
                st.markdown(msg.content, unsafe_allow_html=True)

    # instantiate llamastack connection
    chatClient = LlamaStackClient(base_url=stSession.session_state.api_base_url)

    # create vector db on provider
    chatClient.vector_dbs.register(
        vector_db_id=stSession.session_state.collection_name,
        embedding_model=stSession.session_state.embedding_model_name,
        embedding_dimension=appSettings.config_parameters.vectorstore.embedding_dimensions,
        provider_id=stSession.session_state.provider_name,
    )

    prompt = st.chat_input("💬 Say something...")
    if prompt:
        st.chat_message("user").markdown(prompt)

        # Assistant reply container
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            shield_response = ""

            # execute inference on chat endpoint
            try:
                # inference parameters
                inference_parms = {
                            "max_tokens": stSession.session_state.max_tokens,
                            "n": stSession.session_state.n_comp,
                            "top_p": stSession.session_state.top_p,
                            "temperature": stSession.session_state.temperature,
                            "presence_penalty": stSession.session_state.presence_penalty,
                            "frequency_penalty": stSession.session_state.repeat_penalty
                        }

                # perform rag
                if stSession.session_state.enable_rag:
                    # RAG retrieval call
                    rag_response = chatClient.tool_runtime.rag_tool.query(content=prompt, vector_db_ids=[stSession.session_state.collection_name])

                    # construct the actual prompt to be executed, incorporating the original query and the retrieved content
                    prompt_context = rag_response.content
                    extended_prompt = f"Please answer the given query using the context below.\n\nCONTEXT:\n{prompt_context}\n\nQUERY:\n{prompt}"
                    stSession.session_state.messages.append(UserMessage(content=extended_prompt, role="user"))
                else:
                    stSession.session_state.messages.append(UserMessage(content=prompt, role="user"))

                # if enabled, run safety shield
                if stSession.session_state.enable_shields:
                    shield_output = chatClient.safety.run_shield(
                        messages=[UserMessage(content=prompt, role="user")],
                        shield_id=stSession.session_state.shield_model_name,
                        params={}
                    )

                    # detect unappropriate prompt
                    if shield_output.violation:
                        violation_type = f"Violation Type: {shield_output.violation.metadata.get('violation_type')}"
                        violation_level = f"Violation Level: {shield_output.violation.violation_level}"
                        violation_message = f"Message: {shield_output.violation.user_message}"
                        response_container.markdown(f"**Input Shielding**: {violation_type}, {violation_level} -- {violation_message}")
                    else:
                        # perform inference
                        for line in chatClient.chat.completions.create(messages=stSession.session_state.messages, model=stSession.session_state.model_name, stream=True, **inference_parms):
                            for event in line.choices:
                                token = event.delta.content
                                full_response += token
                            response_container.markdown(full_response, unsafe_allow_html=True)
                else: # no shields
                    # perform inference
                    for line in chatClient.chat.completions.create(messages=stSession.session_state.messages, model=stSession.session_state.model_name, stream=True, **inference_parms):
                        for event in line.choices:
                            token = event.delta.content
                            full_response += token
                        response_container.markdown(full_response, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Request failed: {e}")

            # append full response to chat history
            stSession.session_state.messages.append(CompletionMessage(content=full_response, role="assistant", stop_reason="end_of_message"))

            # save latest messages in the last_chat json file on disk
            stSession.save_chat_history(stSession.session_state.latest_history_filename, stSession.session_state.messages)
