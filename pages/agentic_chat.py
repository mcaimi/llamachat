#!/usr/bin/env python
#
# Agentic Chat. Use llama-stack agentic framework to chat with an AI model,
# with optional features such as tools, shields and rag.
# Streamlit version + llamastack backend
#

import os
import io
import base64
from datetime import datetime
import json, uuid

try:
    import requests
    import streamlit as st
    from dotenv import dotenv_values

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        from libs.shared.settings import Properties
        from libs.shared.session import Session
        from libs.shared.utils import build_header
        from libs.embeddings.embeddings import *
except Exception as e:
    print(f"Caught fatal exception: {e}")

# llama stack
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.agents.react.tool_parser import ReActOutput
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
stSession.add_to_session_state("system_prompt", appSettings.config_parameters.llm.system_prompt)
stSession.add_to_session_state("history_dir", appSettings.config_parameters.openai.history_dir)
stSession.add_to_session_state("latest_history_filename", appSettings.config_parameters.openai.latest_history_filename)
stSession.add_to_session_state("api_key", appSettings.config_parameters.openai.api_key)
stSession.add_to_session_state("agent_messages", [])

# session config
stSession.add_to_session_state("model_name", appSettings.config_parameters.openai.model)
stSession.add_to_session_state("shield_model_name", appSettings.config_parameters.openai.shield_model)
stSession.add_to_session_state("temperature", appSettings.config_parameters.llm.temperature)
stSession.add_to_session_state("top_p", appSettings.config_parameters.llm.top_p)
stSession.add_to_session_state("n_comp", appSettings.config_parameters.llm.n_comp)
stSession.add_to_session_state("max_tokens", appSettings.config_parameters.llm.max_tokens)
stSession.add_to_session_state("presence_penalty", appSettings.config_parameters.llm.presence_penalty)
stSession.add_to_session_state("repeat_penalty", appSettings.config_parameters.llm.repeat_penalty)

# build streamlit UI
st.set_page_config(page_title="🧠 RedHat Agentic AI Assistant", initial_sidebar_state="collapsed", layout="wide")
st.html("assets/header.html")

# instantiate llamastack connection
chatClient = LlamaStackClient(base_url=stSession.session_state.api_base_url)

# Sidebar
with st.sidebar:
    # reset function
    def reset_agent():
        st.cache_resource.clear()
        stSession.remove_from_session_state("agent_session_id")

    st.header("🛠 LLM Control Panel")

    with st.expander("🛠 Settings"):
        model_list = stSession.list_models(model_type="llm")
        stSession.session_state.model_name = st.selectbox(label="Available models", options=model_list, on_change=reset_agent)

        # select operation mode
        agentic_mode = st.radio("Select Agentic Mode", ["**LlamaStack Agentic**", "**ReAct**"],
                                captions=["Use LlamaStack Agentic Framework", "Use Reasoning Model to Improve Tool Calling"],
                                on_change=reset_agent)

        match agentic_mode:
            case "**LlamaStack Agentic**":
                agent_mode = "agentic"
            case "**ReAct**":
                agent_mode = "react"

        enable_persistence = st.checkbox(label="Enable Session Persistence", value=True)

    with st.expander("System Prompt"):
        new_prompt = st.text_area("Update System Prompt", value=stSession.session_state.system_prompt, height=150, on_change=reset_agent,)
        if st.button("🔄 Apply New Prompt"):
            stSession.session_state.system_prompt=new_prompt
            st.success("System prompt updated.")
            reset_agent()

    with st.expander("Model Parameters"):
        stSession.session_state.temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.7, 0.05, on_change=reset_agent,)
        stSession.session_state.top_p = st.slider("📊 Top-P", 0.0, 1.0, 0.9, 0.05, on_change=reset_agent,)
        stSession.session_state.n_comp = st.text_input("Num Completions", value=appSettings.config_parameters.llm.n_comp, on_change=reset_agent,)
        stSession.session_state.max_tokens = st.text_input("🔁 Tokens", value=appSettings.config_parameters.llm.max_tokens, on_change=reset_agent,)
        stSession.session_state.presence_penalty = st.slider("🔁 Presence Penalty", -2.0, 2.0, 1.1, 0.1, on_change=reset_agent,)
        stSession.session_state.repeat_penalty = st.slider("🔁 Repeat Penalty", -2.0, 2.0, 1.1, 0.1, on_change=reset_agent,)

    st.markdown(f"**🔌 Current Endpoint:** `{stSession.session_state.api_base_url}`")
    st.markdown(f"**🔌 Current Model:** `{stSession.session_state.model_name}`")
    st.markdown(f"**🔌 Current Mode:** `{agent_mode}`")

    if st.button("Clear Current Chat"):
        stSession.session_state.agent_messages = []

    st.divider()
    with st.expander("🛠 Agentic"):
        st.markdown("**Shields**")
        shield_models = stSession.list_models(model_type="shield")
        input_shields = st.multiselect(label="Input Shields", options=shield_models, on_change=reset_agent)
        output_shields = st.multiselect(label="Output Shields", options=shield_models, on_change=reset_agent)

        st.markdown("**🔌 Agentic Workflow Capabilities**")
        tool_groups = chatClient.toolgroups.list()
        tool_groups_list = [tool_group.identifier for tool_group in tool_groups]
        mcp_tools_list = [tool for tool in tool_groups_list if tool.startswith("mcp::")]
        builtin_tools_list = [tool for tool in tool_groups_list if not tool.startswith("mcp::")]

        # MCP Servers comes first now
        mcp_label_map = {
            "mcp::tools": "MCP Tools",
            "mcp::opencv": "OpenCV Toolkit",
        }
        mcp_display_options = [mcp_label_map.get(tool, tool) for tool in mcp_tools_list]
        mcp_label_to_tool = {mcp_label_map.get(k, k): k for k in mcp_tools_list}

        st.subheader("MCP Servers")
        mcp_display_selection = st.pills(
            label="Registered APIs",
            options=mcp_display_options,
            selection_mode="multi",
            default=mcp_display_options,
            on_change=reset_agent,
        )
        mcp_selection = [mcp_label_to_tool[label] for label in mcp_display_selection]

        # Builtin Tools
        builtin_label_map = {
            "builtin::websearch": "Web search",
            "builtin::rag": "Retrieval augmented generation",
            "builtin::code_interpreter": "Code interpreter",
            "builtin::wolfram_alpha": "Wolfram Alpha",
        }
        blt_display_options = [builtin_label_map.get(tool, tool) for tool in builtin_tools_list]
        blt_label_to_tool = {builtin_label_map.get(k, k): k for k in builtin_tools_list}

        st.subheader("Builtin Tools")
        blt_display_selection = st.pills(
            label="Registered APIs",
            options=blt_display_options,
            selection_mode="multi",
            on_change=reset_agent,
        )
        toolgroup_selection = [blt_label_to_tool[label] for label in blt_display_selection]

        # if rag is selected, also get a list of all collections in the database
        if "builtin::rag" in toolgroup_selection:
            vector_dbs = chatClient.vector_dbs.list() or []
            if not vector_dbs:
                st.info("No vector databases available for selection. Create one using the Embedding Page.")
                selected_vector_dbs = []
            else:
                vector_dbs = [vector_db.identifier for vector_db in vector_dbs]
                selected_vector_dbs = st.multiselect(
                    label="Select Document Collections to use in RAG queries",
                    options=vector_dbs,
                    on_change=reset_agent,
                )

        # add arguments to tools that need them
        for i, tool_name in enumerate(toolgroup_selection):
            match tool_name:
                case "builtin::rag":
                    tool_dict = dict(
                        name="builtin::rag",
                        args={
                            "vector_db_ids": list(selected_vector_dbs),
                        },
                    )
                    toolgroup_selection[i] = tool_dict
                case "builtin::websearch":
                    tool_dict = dict(
                        name="builtin::websearch",
                        args={
                            "max_results": 10,
                        },
                    )
                    toolgroup_selection[i] = tool_dict

        # Final combined selection
        toolgroup_selection.extend(mcp_selection)

        # display active tools
        active_tool_list = []
        for toolgroup_id in toolgroup_selection:
            if isinstance(toolgroup_id, dict):
                toolgroup_id = toolgroup_id.get("name")

            active_tool_list.extend(
                [
                    f"{''.join(toolgroup_id)}:{t.identifier}"
                    for t in chatClient.tools.list(toolgroup_id=toolgroup_id)
                ]
            )
        with st.expander("🛠 AI Tool Info...", expanded=False):
            st.subheader(f"Active Tools: {len(active_tool_list)}")
            st.json(active_tool_list)

    st.divider()
    with st.expander("💾 Save Chat Log..."):
        save_name = st.text_input("💾 Filename to Save", value=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        if st.button("💾 Save History"):
            stSession.save_chat_history(save_name, stSession.session_state.agent_messages)
            st.success(f"Saved to {save_name}")

        history_files = stSession.list_saved_histories()
        selected_file = st.selectbox("📂 Load History", ["-- Select --"] + history_files)
        if selected_file != "-- Select --" and st.button("📂 Load"):
            stSession.session_state.agent_messages = stSession.load_chat_history(selected_file)
            st.success(f"Loaded {selected_file}")

        if os.path.exists(os.path.join(stSession.session_state.history_dir, stSession.session_state.latest_history_filename)):
            if st.button("🕓 Load Latest Chat"):
                stSession.session_state.agent_messages = stSession.load_chat_history(stSession.session_state.latest_history_filename)
                st.success("Latest chat loaded!")

        if st.button("📤 Export to Markdown"):
            md_filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            stSession.export_chat_to_markdown(md_filename, stSession.session_state.agent_messages)
            st.success(f"Exported to {md_filename}")

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

# Define Agent for AI Interaction
@st.cache_resource
def instantiate_ai_agent(model_name, sysPrompt, availableTools, inferenceParms):
    match agent_mode:
        case "agentic":
            return Agent(
                chatClient, 
                model=model_name,
                instructions=f"""{sysPrompt}. You have tools available that you can use to respond to the user.""" ,
                tools=availableTools,
                tool_config={"tool_choice":"auto"},
                sampling_params=inferenceParms,
                # Configure safety (optional)
                input_shields=input_shields,
                output_shields=output_shields,
                enable_session_persistence=enable_persistence,
            )
        case "react":
            return ReActAgent(
                chatClient, 
                model=model_name,
                instructions=f"""{sysPrompt}. You have tools available that you can use to respond to the user.""" ,
                tools=availableTools,
                response_format={
                    "type": "json_schema",
                    "json_schema": ReActOutput.model_json_schema(),
                },
                tool_config={"tool_choice":"auto"},
                sampling_params=inferenceParms,
                input_shields=input_shields,
                output_shields=output_shields,
                enable_session_persistence=enable_persistence,
            )
chatAgent = instantiate_ai_agent(stSession.session_state.model_name,
                                stSession.session_state.system_prompt,
                                toolgroup_selection, inference_parms)

# create local agent session to maintain memory
stSession.add_to_session_state("agent_session_id", chatAgent.create_session(session_name=f"agent_session_{uuid.uuid4()}"))

# Chat Interface
for msg in stSession.session_state.agent_messages:
    if msg.role != "system":
        with st.chat_message(msg.role):
            st.markdown(msg.content, unsafe_allow_html=True)

prompt_raw = st.chat_input(placeholder="💬 Say something...", accept_file=True, file_type=appSettings.config_parameters.features.supported_img_formats + appSettings.config_parameters.features.supported_data_formats)
if prompt_raw:
    prompt = prompt_raw.get('text')
    uploaded_files = prompt_raw.get('files')
    st.chat_message("user").markdown(prompt)

    # Assistant reply container
    with st.chat_message("assistant"):
        # execute inference on chat endpoint
        try:
            augmented_prompt = f"{prompt}."

            # if the user specifies a file, then we need to process it
            if len(uploaded_files) > 0:
                for f in uploaded_files:
                    if f.name.split(".")[-1] in appSettings.config_parameters.features.supported_img_formats:
                        st.image(f)
                        
                        # base64 encoding
                        im_b64 = base64.b64encode(f.read()).decode("utf-8")

                        # update prompt:
                        augmented_prompt = []
                        augmented_prompt.append({"type": "text", "text": f"{prompt}."})
                        augmented_prompt.append({
                            "type": "image",
                            "image": {
                                "data": im_b64
                            }
                        })
                    else:
                        with st.spinner(f"🧠 Embedding.... {f.name}"):
                            # instantiate converter
                            converter = createDoclingConverter(do_ocr=False, do_table_structure=True)
                            # prepare documents to be embedded
                            st.markdown(f"**Prepare Document...**")
                            docs = prepareDocuments(converter, uploaded_files=[f])
                            augmented_query = ""
                            for d in docs:
                                augmented_query += d.get('doc').document.export_to_markdown()
                            
                            # update prompt...
                            augmented_prompt += f"Your context is: {augmented_query}"

                            del converter
                        st.markdown("** Conversion Done! **")

            # append user request
            stSession.session_state.agent_messages.append(UserMessage(content=prompt, role="user"))

            # chat with the ai agent
            with st.spinner("🧠Thinking...."):
                response = chatAgent.create_turn(
                    messages=[{"role": "user", "content": augmented_prompt}],
                    session_id=stSession.session_state.agent_session_id,
                    stream=True
                )

            # parse responses
            message_placeholder = st.empty()
            full_response = ""
            retrieval_response = ""
            for log in AgentEventLogger().log(response):

                if log.role == "tool_execution":
                    retrieval_response += log.content.replace("====", "").strip()
                else:
                    full_response += log.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            
            with st.expander("Tool Call Info"):
                retrieval_message_placeholder = st.empty()
                retrieval_message_placeholder.markdown(retrieval_response)
        except Exception as e:
            st.error(f"Request failed: {e}")

        # add to history
        stSession.session_state.agent_messages.append(CompletionMessage(role="assistant", content=full_response, stop_reason="end_of_turn"))

        # save latest messages in the last_chat json file on disk
        stSession.save_chat_history(stSession.session_state.latest_history_filename, stSession.session_state.agent_messages)
