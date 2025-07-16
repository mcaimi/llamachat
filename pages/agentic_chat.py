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
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from llama_stack_client.types import UserMessage, SystemMessage, CompletionMessage
from llama_stack_client.types.shared.content_delta import TextDelta, ToolCallDelta

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# initialize streamlit session
stSession = Session(st.session_state)

# initialize streamlit session
stSession = Session(st.session_state)
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
stSession.add_to_session_state("messages", [])

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
st.set_page_config(page_title="🧠 RedHat Agentic AI Assistant", initial_sidebar_state="collapsed", layout="wide")
st.html("assets/header.html")

# instantiate llamastack connection
chatClient = LlamaStackClient(base_url=stSession.session_state.api_base_url)

# Sidebar
with st.sidebar:
    st.header("🛠 LLM Control Panel")

    if st.button("Clear Current Chat"):
        stSession.session_state.messages = []

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

    st.markdown("**🔌 Agentic Workflow Capabilities**")
    available_tools = []
    enabled_tools = []
    for tool in chatClient.tool_runtime.list_tools():
        available_tools.append(f"{tool.toolgroup_id}/{tool.identifier}")
    enabled_tools = st.pills("Available Tools", available_tools, selection_mode="multi")

# Chat Interface
for msg in stSession.session_state.messages:
    if msg.role != "system":
        with st.chat_message(msg.role):
            st.markdown(msg.content, unsafe_allow_html=True)

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
            # create vector db on provider
            chatClient.vector_dbs.register(
                vector_db_id=stSession.session_state.collection_name,
                embedding_model=stSession.session_state.embedding_model_name,
                embedding_dimension=appSettings.config_parameters.vectorstore.embedding_dimensions,
                provider_id=stSession.session_state.provider_name,
            )

            # inference parameters
            inference_parms = {
                        "max_tokens": stSession.session_state.max_tokens,
                        "n": stSession.session_state.n_comp,
                        "strategy": {
                            "type": "top_p",
                            "temperature": stSession.session_state.temperature,
                            "top_p": stSession.session_state.top_p,
                            },
                        "presence_penalty": stSession.session_state.presence_penalty,
                        "frequency_penalty": stSession.session_state.repeat_penalty
                    }

            chatAgent = Agent(
                chatClient, 
                model=stSession.session_state.model_name,
                instructions=f"""{stSession.session_state.system_prompt}. You have tools available that can be used to respond to the user.
                            """ ,
                tools=[
                    {
                        "name": "builtin::rag/knowledge_search",
                        "args": {"vector_db_ids": [stSession.session_state.collection_name]},
                    }
                ],
                sampling_params=inference_parms
            )

            # append user request
            stSession.session_state.messages.append(UserMessage(content=prompt, role="user"))

            # chat with the ai agent
            session_id = chatAgent.create_session("web-session")
            response = chatAgent.create_turn(
                messages=stSession.session_state.messages,
                session_id=session_id,
                stream=True
            )

            # parse responses
            retrieval_message_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            retrieval_response = ""
            for log in AgentEventLogger().log(response):
                log.print()
                if log.role == "tool_execution":
                    retrieval_response += log.content.replace("====", "").strip()
                    retrieval_message_placeholder.info(retrieval_response)
                else:
                    full_response += log.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            # remove agent
            del chatAgent
        except Exception as e:
            st.error(f"Request failed: {e}")

        # save latest messages in the last_chat json file on disk
        stSession.save_chat_history(stSession.session_state.latest_history_filename, stSession.session_state.messages)
