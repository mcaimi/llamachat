#!/usr/bin/env python
#
# Agentic Chat. Use llama-stack agentic framework to chat with an AI model,
# with optional features such as tools, shields and rag.
# Streamlit version + llamastack backend
#

import os
import base64
from datetime import datetime
import uuid

try:
    import streamlit as st
    from dotenv import dotenv_values

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        from libs.shared.settings import Properties
        from libs.shared.session import Session
        from libs.shared.agent import Agent, AgentSession
        from libs.shared.state import AgentMessage
        from libs.shared.responses import format_response
        from libs.embeddings.embeddings import *
except Exception as e:
    print(f"Caught fatal exception: {e}")

# llama stack
from llama_stack_client import LlamaStackClient

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# initialize streamlit session
stSession = Session(st.session_state)

# setup default values from config file
stSession.add_to_session_state(
    "api_base_url", appSettings.config_parameters.openai.default_local_api
)
stSession.add_to_session_state(
    "system_prompt", appSettings.config_parameters.llm.system_prompt
)
stSession.add_to_session_state(
    "history_dir", appSettings.config_parameters.openai.history_dir
)
stSession.add_to_session_state(
    "latest_history_filename",
    appSettings.config_parameters.openai.latest_history_filename,
)
stSession.add_to_session_state("api_key", appSettings.config_parameters.openai.api_key)
stSession.add_to_session_state("agent_messages", [])

# session config
stSession.add_to_session_state("model_name", appSettings.config_parameters.openai.model)
stSession.add_to_session_state(
    "shield_model_name", appSettings.config_parameters.openai.shield_model
)
stSession.add_to_session_state(
    "temperature", appSettings.config_parameters.llm.temperature
)
stSession.add_to_session_state("max_infer_iters", appSettings.config_parameters.llm.max_infer_iters)
stSession.add_to_session_state("max_tool_calls", appSettings.config_parameters.llm.max_tool_calls)
stSession.add_to_session_state("parallel_tool_calls", appSettings.config_parameters.llm.parallel_tool_calls)
stSession.add_to_session_state(
    "max_output_tokens", appSettings.config_parameters.llm.max_output_tokens
)
stSession.add_to_session_state("timeout", appSettings.config_parameters.llm.timeout)

# build streamlit UI
st.set_page_config(
    page_title="🧠 Agentic AI Assistant",
    initial_sidebar_state="collapsed",
    layout="wide",
)
st.html("assets/header.html")

# instantiate llamastack connection
chatClient = LlamaStackClient(base_url=stSession.session_state.api_base_url)

# Sidebar
with st.sidebar:
    # reset function
    def reset_agent():
        st.cache_resource.clear()

    st.header("🛠 LLM Control Panel")

    with st.expander("🛠 Settings"):
        model_list = stSession.list_models(model_type="llm")
        stSession.session_state.model_name = st.selectbox(
            label="Available models", options=model_list, on_change=reset_agent
        )

        # select operation mode
        agentic_mode = st.radio(
            "Select Interaction Mode",
            ["**LLM Chat**", "**Agentic**"],
            captions=[
                "Chat with a model to answer questions and generate text.",
                "Interact with tools and MCP servers.",
            ],
            on_change=reset_agent,
        )

        match agentic_mode:
            case "**LLM Chat**":
                agent_mode = "chat"
            case "**Agentic**":
                agent_mode = "agent"

    with st.expander("System Prompt"):
        new_prompt = st.text_area(
            "Update System Prompt",
            value=stSession.session_state.system_prompt,
            height=150,
            on_change=reset_agent,
        )
        if st.button("🔄 Apply New Prompt"):
            stSession.update_system_prompt(new_prompt)
            st.success("System prompt updated.")
            reset_agent()

    with st.expander("Model Parameters"):
        stSession.session_state.temperature = st.slider(
            "🌡️ Temperature",
            0.0,
            2.0,
            appSettings.config_parameters.llm.temperature,
            0.05,
            on_change=reset_agent,
        )
        stSession.session_state.max_output_tokens = st.number_input(
            "🔁 Tokens",
            min_value = 16,
            value=appSettings.config_parameters.llm.max_output_tokens,
            on_change=reset_agent,
        )
        stSession.session_state.max_infer_iters = st.number_input(
            "🔁 Max Inference Iterations",
            min_value=1,
            max_value=100,
            value=appSettings.config_parameters.llm.max_infer_iters,
            on_change=reset_agent,
        )
        stSession.session_state.max_tool_calls = st.number_input(
            "Max Number of Tool Calls",
            min_value=1,
            max_value=100,
            value=appSettings.config_parameters.llm.max_tool_calls,
            on_change=reset_agent,
        )
        stSession.session_state.parallel_tool_calls = st.checkbox(
            "Enable Parallel Tool Calls",
            value=appSettings.config_parameters.llm.parallel_tool_calls,
            on_change=reset_agent,
        )
        stSession.session_state.timeout = st.number_input(
            "Inference Timeout",
            min_value=30,
            max_value=500,
            value=appSettings.config_parameters.llm.timeout,
            on_change=reset_agent,
        )

    st.markdown(f"**🔌 Current Endpoint:** `{stSession.session_state.api_base_url}`")
    st.markdown(f"**🔌 Current Model:** `{stSession.session_state.model_name}`")
    st.markdown(f"**🔌 Current Mode:** `{agent_mode}`")

    if st.button("Reset Agent State"):
        stSession.clear_chat_session()
        reset_agent()

    st.divider()
    with st.expander("🛠 Advanced"):
        st.markdown("**Shields**")
        shield_models = stSession.list_models(model_type="shield")

        # select input shields
        in_shield_objects = st.multiselect(
            label="Input Shields", options=[m['id'] for m in shield_models], on_change=reset_agent
        )
        input_shields = [m['model'] for m in shield_models if m['id'] in in_shield_objects]

        # select output shields
        out_shield_objects = st.multiselect(
            label="Output Shields", options=[m['id'] for m in shield_models], on_change=reset_agent
        )
        output_shields = [m['model'] for m in shield_models if m['id'] in out_shield_objects]


        # if mode is Agent...
        if agent_mode == "agent":
            st.markdown("**🔌 Agentic Workflow Capabilities**")
            # get a list of toolgroups * DEPRECATED API *
            tool_groups = chatClient.toolgroups.list()

            # build list of available MCP endpoints
            mcp_tools_list = [
                {
                    "type": "mcp",
                    "server_url": tool.mcp_endpoint.uri,
                    "server_label": tool.identifier
                }
                for tool in tool_groups if tool.identifier.startswith("mcp::")
            ]
            
            # MCP Servers comes first now
            st.subheader("MCP Servers")
            mcp_selection = st.pills(
                label="Registered APIs",
                options=[t.get("server_label") for t in mcp_tools_list],
                default=[t.get("server_label") for t in mcp_tools_list],
                selection_mode="multi",
                on_change=reset_agent,
            )

            # Final combined selection
            toolgroup_selection = []
            toolgroup_selection.extend([tool for tool in mcp_tools_list if tool.get("server_label") in mcp_selection])

            # rag capability
            enable_rag = st.checkbox(
                "Enable RAG",
                value=False,
                on_change=reset_agent
            )

            # display available vector ids
            vector_ids = st.multiselect(
                "Select Vector Databases",
                options=[vector_db.name for vector_db in chatClient.vector_stores.list()],
                disabled=not enable_rag
            )

            if enable_rag:
                toolgroup_selection.extend([{
                        "type": "file_search",
                        "vector_store_ids": [v.id for v in chatClient.vector_stores.list() if v.name in vector_ids] or [],
                    }]
                )

            # display active tools
            active_tool_list = []
            for toolgroup_id in toolgroup_selection:
                if isinstance(toolgroup_id, dict):
                    tool_type = toolgroup_id.get("type")

                    match tool_type:
                        case "mcp":
                            toolgroup_id = toolgroup_id.get("server_label")

                            active_tool_list.extend(
                                [
                                    f"{''.join(toolgroup_id)}:{t.name}"
                                    for t in chatClient.tools.list(toolgroup_id=toolgroup_id)
                                ]
                            )
                        case "file_search":
                            vector_ids = toolgroup_id.get("vector_store_ids")
                            active_tool_list.extend(
                                [
                                    f"inline::file_search::{vector_ids}"
                                ]
                            )

            with st.expander("🛠 AI Tool Info...", expanded=False):
                st.subheader(f"Active Tools: {len(active_tool_list)}")
                st.json(active_tool_list)
        else:
            toolgroup_selection = None


    st.divider()
    with st.expander("💾 Save Chat Log..."):
        save_name = st.text_input(
            "💾 Filename to Save",
            value=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        if st.button("💾 Save History"):
            stSession.save_chat_history(
                save_name, stSession.session_state.agent_messages
            )
            st.success(f"Saved to {save_name}")

        history_files = stSession.list_saved_histories()
        selected_file = st.selectbox(
            "📂 Load History", ["-- Select --"] + history_files
        )
        if selected_file != "-- Select --" and st.button("📂 Load"):
            stSession.session_state.agent_messages = stSession.load_chat_history(
                selected_file
            )
            st.success(f"Loaded {selected_file}")

        if os.path.exists(
            os.path.join(
                stSession.session_state.history_dir,
                stSession.session_state.latest_history_filename,
            )
        ):
            if st.button("🕓 Load Latest Chat"):
                stSession.session_state.agent_messages = stSession.load_chat_history(
                    stSession.session_state.latest_history_filename
                )
                st.success("Latest chat loaded!")

        if st.button("📤 Export to Markdown"):
            md_filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            stSession.export_chat_to_markdown(
                md_filename, stSession.session_state.agent_messages
            )
            st.success(f"Exported to {md_filename}")

# inference parameters
inference_parms = {
    #"max_output_tokens": int(stSession.session_state.max_output_tokens),
    "temperature": float(stSession.session_state.temperature),
    "timeout": int(stSession.session_state.timeout),
    "max_infer_iters": int(stSession.session_state.max_infer_iters),
    "max_tool_calls": int(stSession.session_state.max_tool_calls),
    "parallel_tool_calls": bool(stSession.session_state.parallel_tool_calls),
}

# Define Agent for AI Interaction
@st.cache_resource
def instantiate_ai_agent(_client,
                        model_name,
                        instructions,
                        tools,
                        parameters,
                        input_shields,
                        output_shields,
                        inferenceParms):
    match agent_mode:
        case "chat":
            return Agent(
                llamastack_client = _client,
                model = model_name,
                instructions = f"""{instructions}.""",
                input_shields=input_shields,
                output_shields=output_shields,
                sampling_params=inferenceParms,
            )
        case "agent":
            return Agent(
                llamastack_client = _client,
                model = model_name,
                instructions = f"""{instructions}. You have tools available that you can use to respond to the user.""",
                tools=tools,
                input_shields=input_shields,
                output_shields=output_shields,
                sampling_params=inferenceParms,
            )

chatAgent = instantiate_ai_agent(
    _client = chatClient,
    model_name = stSession.session_state.model_name,
    instructions = stSession.session_state.system_prompt,
    tools = toolgroup_selection,
    parameters = inference_parms,
    input_shields=input_shields,
    output_shields=output_shields,
    inferenceParms=inference_parms
)

# Chat Interface
for msg in stSession.session_state.agent_messages:
    if msg.role != "system":
        with st.chat_message(msg.role):
            st.markdown(msg.content, unsafe_allow_html=True)

prompt_raw = st.chat_input(
    placeholder="💬 Say something...",
    accept_file=True,
    file_type=appSettings.config_parameters.features.supported_img_formats
    + appSettings.config_parameters.features.supported_data_formats,
)
if prompt_raw:
    prompt = prompt_raw.get("text")
    uploaded_files = prompt_raw.get("files")
    st.chat_message("user").markdown(prompt)

    # Assistant reply container
    with st.chat_message("assistant"):
        # execute inference on chat endpoint
        try:
            augmented_prompt = f"{prompt}."

            # if the user specifies a file, then we need to process it
            if len(uploaded_files) > 0:
                for f in uploaded_files:
                    if (
                        f.name.split(".")[-1]
                        in appSettings.config_parameters.features.supported_img_formats
                    ):
                        st.image(f)

                        # base64 encoding
                        im_b64 = base64.b64encode(f.read()).decode("utf-8")
                        # image entity in content
                        img_entity = {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{im_b64}",
                        }
                        # text entity_in content
                        txt_entity = {
                            "type": "input_text",
                            "text": f"{prompt}",
                        }

                        # update prompt:
                        augmented_prompt = [{
                            "role": "user",
                            "content": [
                                txt_entity,
                                img_entity
                            ]
                        }]
                    else:
                        with st.spinner(f"🧠 Creating Docling Converter.... {f.name}"):
                            # instantiate converter
                            converter = createDoclingConverter(
                                do_ocr=False, do_table_structure=True
                            )
                            # prepare documents to be embedded
                            st.markdown(f"**Prepare Document...**")
                            docs = prepareDocuments(converter, uploaded_files=[f])
                            augmented_query = ""
                            for d in docs:
                                augmented_query += d.get(
                                    "doc"
                                ).document.export_to_markdown()

                            # update prompt...
                            augmented_prompt += f"What follows is the context you have to use to answer the question: {augmented_query}"

                            del converter
                        st.markdown("** Conversion Done! **")

            # append user request
            stSession.session_state.agent_messages.append(
                AgentMessage(_content=prompt, _role="user")
            )

            # run input shield...
            with st.spinner("Running Input Shield..."):
                flagged, output = chatAgent.input_shield(augmented_prompt)

            if not flagged:
                # chat with the ai agent
                with st.spinner("🧠Thinking...."):
                    response = chatAgent.create_turn(
                        prompt=augmented_prompt
                    )

                # parse responses
                message_placeholder = st.empty()
                prompt_response, tool_response = format_response(response)

                # perform output shielding...
                message_placeholder = st.empty()
                with st.spinner("Running Output Shield..."):
                    flagged, output = chatAgent.output_shield(prompt_response)

                if flagged:
                    # filter response
                    prompt_response = f"Shield Active: {output.str()}. Cannot perform inference."
                
                message_placeholder.markdown(prompt_response)    
                
                with st.expander("Inference Stack"):
                    retrieval_message_placeholder = st.empty()
                    retrieval_message_placeholder.markdown(tool_response)
            else:
                prompt_response = f"Shield Active: {output.str()}. Cannot perform inference."
                message_placeholder = st.empty()
                message_placeholder.markdown(prompt_response)
            
        except Exception as e:
            st.error(f"Request failed: {e}")

        # add to history
        stSession.session_state.agent_messages.append(
            AgentMessage(
                _role="assistant", _content=prompt_response
            )
        )

        # save latest messages in the last_chat json file on disk
        stSession.save_chat_history(
            stSession.session_state.latest_history_filename,
            stSession.session_state.agent_messages,
        )
