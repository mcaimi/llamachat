#!/usr/bin/env python
#
# Agentic Chat. Use ogx agentic framework to chat with an AI model,
# with optional features such as tools, and rag.
# Streamlit version + ogx backend
#

import os
import base64
from datetime import datetime

try:
    import streamlit as st
    from dotenv import dotenv_values

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        from libs.shared.settings import Properties
        from libs.shared.session import Session
        from libs.shared.agent import Agent
        from libs.shared.state import AgentMessage
        from libs.shared.responses import format_response, format_streaming_response
        from libs.embeddings.embeddings import *
        from libs.agents.loader import discover_agents, load_agent_definition
        from libs.agents.definition import AgentDefinition
        from libs.skills.loader import discover_skills, activate_skill
        from libs.agents.tools import (
            SHELL_TOOL, run_command, extract_function_calls, parse_call_arguments,
        )
except Exception as e:
    print(f"Caught fatal exception: {e}")

# OGX
from ogx_client import OgxClient

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
stSession.add_to_session_state("selected_agent_name", AgentDefinition.NONE_AGENT_NAME)
stSession.add_to_session_state("active_skills", [])
stSession.add_to_session_state("pending_tool_execution", None)

# session config
stSession.add_to_session_state("model_name", appSettings.config_parameters.openai.model)
stSession.add_to_session_state(
    "temperature", appSettings.config_parameters.llm.temperature
)
stSession.add_to_session_state(
    "max_infer_iters", appSettings.config_parameters.llm.max_infer_iters
)
stSession.add_to_session_state(
    "max_tool_calls", appSettings.config_parameters.llm.max_tool_calls
)
stSession.add_to_session_state(
    "parallel_tool_calls", appSettings.config_parameters.llm.parallel_tool_calls
)
stSession.add_to_session_state(
    "max_output_tokens", appSettings.config_parameters.llm.max_output_tokens
)
stSession.add_to_session_state("timeout", appSettings.config_parameters.llm.timeout)
stSession.add_to_session_state("stream", appSettings.config_parameters.openai.stream)

# discover available agents and skills
@st.cache_data(ttl=60)
def _cached_discover_agents(paths_tuple):
    return discover_agents(list(paths_tuple))

@st.cache_data(ttl=60)
def _cached_discover_skills(paths_tuple):
    return discover_skills(list(paths_tuple))

_agent_cfg = getattr(appSettings.config_parameters, "agents", None)
_agent_paths = _agent_cfg.paths if _agent_cfg and hasattr(_agent_cfg, "paths") else ["~/.config/opencode/agents"]
_skill_cfg = getattr(appSettings.config_parameters, "skills", None)
_skill_paths = _skill_cfg.paths if _skill_cfg and hasattr(_skill_cfg, "paths") else ["~/.config/opencode/skills"]

available_agents = _cached_discover_agents(tuple(_agent_paths))
available_skills = _cached_discover_skills(tuple(_skill_paths))

# build streamlit UI
st.set_page_config(
    page_title="🧠 Agentic AI Assistant",
    initial_sidebar_state="collapsed",
    layout="wide",
)
st.html("assets/header.html")

# instantiate ogx connection
chatClient = OgxClient(base_url=stSession.session_state.api_base_url)

# Sidebar
with st.sidebar:
    # reset function
    def reset_agent():
        st.cache_resource.clear()

    st.header("🛠 LLM Control Panel")

    with st.expander("🛠 Settings"):
        try:
            model_list = [
                m.id for m in chatClient.models.list()
                if m.custom_metadata and m.custom_metadata.get("model_type") == "llm"
            ]
        except Exception:
            model_list = []
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

        if agent_mode == "agent" and available_agents:
            agent_options = [AgentDefinition.NONE_AGENT_NAME] + [
                a.name for a in available_agents
            ]
            stSession.session_state.selected_agent_name = st.selectbox(
                label="Agent Profile",
                options=agent_options,
                on_change=reset_agent,
                help="Select a predefined agent profile to override prompt, tools, and parameters.",
            )

        stream = st.checkbox(
            label="Stream Responses", value=stSession.session_state.stream
        )

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
            min_value=16,
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
    if agent_mode == "agent":
        st.markdown(f"**🔌 Current Agent:** `{stSession.session_state.selected_agent_name}`")

    if st.button("Reset Agent State"):
        stSession.clear_chat_session()
        reset_agent()

    st.divider()
    with st.expander("🛠 Advanced"):
        # if mode is Agent...
        if agent_mode == "agent":
            st.markdown("**🔌 Agentic Workflow Capabilities**")
            connectors = chatClient.connectors.connectors_v1alpha_admin_connectors_get()

            # build list of available MCP endpoints
            mcp_tools_list = [
                tool for tool in connectors if tool.connector_id.startswith("mcp::")
            ]

            # MCP Servers comes first now
            st.subheader("MCP Servers")
            mcp_selection = st.pills(
                label="Registered APIs",
                options=[t.server_label for t in mcp_tools_list],
                default=[t.server_label for t in mcp_tools_list],
                selection_mode="multi",
                on_change=reset_agent,
            )

            # Final combined selection
            toolgroup_selection = []
            toolgroup_selection.extend(
                [
                    {
                        "type": "mcp",
                        "server_url": tool.url,
                        "server_label": tool.server_label,
                    }
                    for tool in mcp_tools_list
                    if tool.server_label in mcp_selection
                ]
            )

            # rag capability
            enable_rag = st.checkbox("Enable RAG", value=False, on_change=reset_agent)

            # display available vector ids
            vector_ids = st.multiselect(
                "Select Vector Databases",
                options=[
                    vector_db.name for vector_db in chatClient.vector_stores.list()
                ],
                disabled=not enable_rag,
            )

            if enable_rag:
                toolgroup_selection.extend(
                    [
                        {
                            "type": "file_search",
                            "vector_store_ids": [
                                v.id
                                for v in chatClient.vector_stores.list()
                                if v.name in vector_ids
                            ]
                            or [],
                        }
                    ]
                )

            # shell/command execution tool (enabled by default in agent mode)
            enable_shell = st.checkbox(
                "Enable Command Execution", value=True, on_change=reset_agent
            )
            if enable_shell:
                toolgroup_selection.append(SHELL_TOOL)

            # discover tools from selected connectors
            active_tool_list = []
            for connector in mcp_tools_list:
                if connector.server_label in mcp_selection:
                    try:
                        connector_tools = chatClient.connectors.connector_tools_v1alpha_admin_connectors_connector_id_tools_get(
                            connector_id=connector.connector_id
                        )
                        active_tool_list.extend(
                            [f"{connector.server_label}:{t.name}" for t in connector_tools]
                        )
                    except Exception:
                        active_tool_list.append(f"mcp:{connector.server_label}")

            with st.expander("🛠 AI Tool Info...", expanded=False):
                st.subheader(f"Active Tools: {len(active_tool_list)}")
                st.json(active_tool_list)
        else:
            st.markdown("Agentic Features Disabled.")
            toolgroup_selection = None

    if agent_mode == "agent" and available_skills:
        st.divider()
        with st.expander("Skills"):
            st.markdown("Skills augment the agent's knowledge. Select skills to activate.")
            skill_selection = st.pills(
                label="Available Skills",
                options=[s.name for s in available_skills],
                default=[],
                selection_mode="multi",
                on_change=reset_agent,
            )
            stSession.session_state.active_skills = skill_selection or []

            for s in available_skills:
                st.caption(f"**{s.name}**: {s.description}")

    st.divider()
    with st.expander("💾 Save Chat Log..."):
        save_name = st.text_input(
            "💾 Filename to Save",
            value=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        if st.button("💾 Save History"):
            try:
                saved_path = stSession.save_chat_history(
                    save_name, stSession.session_state.agent_messages
                )
                st.success(f"Saved to {saved_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

        try:
            history_files = stSession.list_saved_histories()
        except Exception as e:
            history_files = []
            st.error(f"Could not list histories: {e}")
        selected_file = st.selectbox(
            "📂 Load History", ["-- Select --"] + history_files
        )
        if selected_file != "-- Select --" and st.button("📂 Load"):
            try:
                stSession.session_state.agent_messages = stSession.load_chat_history(
                    selected_file
                )
                st.success(f"Loaded {selected_file}")
            except Exception as e:
                st.error(f"Load failed: {e}")

        latest_candidate_path = os.path.join(
            stSession.session_state.history_dir,
            os.path.basename(stSession.session_state.latest_history_filename),
        )
        if os.path.exists(latest_candidate_path):
            if st.button("🕓 Load Latest Chat"):
                try:
                    stSession.session_state.agent_messages = (
                        stSession.load_chat_history(
                            stSession.session_state.latest_history_filename
                        )
                    )
                    st.success("Latest chat loaded!")
                except Exception as e:
                    st.error(f"Load latest failed: {e}")

        if st.button("📤 Export to Markdown"):
            md_filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            try:
                exported_path = stSession.export_chat_to_markdown(
                    md_filename, stSession.session_state.agent_messages
                )
                st.success(f"Exported to {exported_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")

# resolve agent definition overrides
active_agent_def = None
if (
    agent_mode == "agent"
    and stSession.session_state.selected_agent_name != AgentDefinition.NONE_AGENT_NAME
):
    matching = [
        a
        for a in available_agents
        if a.name == stSession.session_state.selected_agent_name
    ]
    if matching:
        active_agent_def = load_agent_definition(matching[0].directory)

        if active_agent_def.model_parameters:
            params = active_agent_def.model_parameters
            if "temperature" in params:
                stSession.session_state.temperature = params["temperature"]
            if "max_output_tokens" in params:
                stSession.session_state.max_output_tokens = params["max_output_tokens"]
            if "max_infer_iters" in params:
                stSession.session_state.max_infer_iters = params["max_infer_iters"]
            if "max_tool_calls" in params:
                stSession.session_state.max_tool_calls = params["max_tool_calls"]
            if "parallel_tool_calls" in params:
                stSession.session_state.parallel_tool_calls = params["parallel_tool_calls"]
            if "timeout" in params:
                stSession.session_state.timeout = params["timeout"]

        stSession.session_state.system_prompt = active_agent_def.to_system_prompt()

        if active_agent_def.preferred_model and active_agent_def.preferred_model in model_list:
            stSession.session_state.model_name = active_agent_def.preferred_model

        if active_agent_def.mcp_servers and agent_mode == "agent":
            available_labels = [t.server_label for t in mcp_tools_list]
            filtered_labels = [s for s in active_agent_def.mcp_servers if s in available_labels]
            toolgroup_selection = [
                {"type": "mcp", "server_url": tool.url, "server_label": tool.server_label}
                for tool in mcp_tools_list
                if tool.server_label in filtered_labels
            ]

        if active_agent_def.rag and active_agent_def.rag.get("enabled"):
            enable_rag = True

# activate selected skills and augment system prompt
activated_skill_instructions = []
activated_skill_defs = []
all_active_skill_names = set(stSession.session_state.active_skills)
if active_agent_def and active_agent_def.skills:
    all_active_skill_names.update(active_agent_def.skills)

for skill_name in all_active_skill_names:
    matching_skills = [s for s in available_skills if s.name == skill_name]
    if matching_skills:
        skill_def = activate_skill(matching_skills[0].directory)
        activated_skill_defs.append(skill_def)
        resolved_instructions = skill_def.instructions.replace(
            "{{SKILL_DIR}}", skill_def.directory
        )
        activated_skill_instructions.append(
            f"## Skill: {skill_def.name}\n\n{resolved_instructions}"
        )

# enforce allowed_tools from active skills
if agent_mode == "agent" and activated_skill_defs:
    all_allowed = set()
    has_restriction = False
    for skill_def in activated_skill_defs:
        if skill_def.allowed_tools:
            has_restriction = True
            for tool_name in skill_def.allowed_tools.split(","):
                all_allowed.add(tool_name.strip())
    if has_restriction and toolgroup_selection:
        toolgroup_selection = [
            t for t in toolgroup_selection
            if (isinstance(t, dict) and t.get("type") in ("mcp", "file_search"))
            or (isinstance(t, dict) and t.get("name") in all_allowed)
        ]

effective_instructions = stSession.session_state.system_prompt

if agent_mode == "agent" and available_skills:
    skill_catalog = "\n".join(
        f"- **{s.name}**: {s.description}" for s in available_skills
    )
    effective_instructions += (
        "\n\n---\n\n## Available Skills\n\n"
        "The following skills are available and can be activated by the user:\n\n"
        + skill_catalog
    )

if activated_skill_instructions:
    effective_instructions += (
        "\n\n---\n\n## Active Skills\n\n"
        "The following skills are active. Follow their guidance:\n\n"
        + "\n\n".join(activated_skill_instructions)
    )

# inference parameters
inference_parms = {
    # "max_output_tokens": int(stSession.session_state.max_output_tokens),
    "temperature": float(stSession.session_state.temperature),
    "timeout": int(stSession.session_state.timeout),
    "max_infer_iters": int(stSession.session_state.max_infer_iters),
    "max_tool_calls": int(stSession.session_state.max_tool_calls),
    "parallel_tool_calls": bool(stSession.session_state.parallel_tool_calls),
}


# Define Agent for AI Interaction
@st.cache_resource
def instantiate_ai_agent(
    _client, model_name, instructions, tools, parameters, inferenceParms
):
    match agent_mode:
        case "chat":
            return Agent(
                ogx_client=_client,
                model=model_name,
                instructions=f"""{instructions}.""",
                sampling_params=inferenceParms,
            )
        case "agent":
            return Agent(
                ogx_client=_client,
                model=model_name,
                instructions=f"""{instructions}. You have tools available that you can use to respond to the user.""",
                tools=tools,
                sampling_params=inferenceParms,
            )


chatAgent = instantiate_ai_agent(
    _client=chatClient,
    model_name=stSession.session_state.model_name,
    instructions=effective_instructions,
    tools=toolgroup_selection,
    parameters=inference_parms,
    inferenceParms=inference_parms,
)

# -- helper: process a response and check for local function calls --
def _process_response_for_function_calls(response):
    """Check response for execute_command calls. Returns (function_calls, text_response, tool_response, tool_outputs)."""
    function_calls = extract_function_calls(response)
    local_calls = [fc for fc in function_calls if fc["name"] == "execute_command"]
    prompt_response, tool_response, tool_outputs = format_response(response)
    return local_calls, prompt_response, tool_response, tool_outputs


def _process_stream_for_function_calls(stream_iter):
    """Consume a streaming response. Returns (final_response, function_calls, text, callstack, tool_outputs)."""
    prompt_response = ""
    callstack_response = ""
    tool_outputs = []
    final_response = None
    message_placeholder = st.empty()

    for item in stream_iter:
        stream_fragment, callstack_fragment, tool_output = format_streaming_response(item)
        prompt_response += stream_fragment
        callstack_response += callstack_fragment
        if tool_output:
            tool_outputs.append(tool_output)
        message_placeholder.markdown(prompt_response)
        if hasattr(item, "type") and item.type == "response.completed":
            final_response = item.response

    local_calls = []
    if final_response:
        all_calls = extract_function_calls(final_response)
        local_calls = [fc for fc in all_calls if fc["name"] == "execute_command"]

    return final_response, local_calls, prompt_response, callstack_response, tool_outputs


# Chat Interface
for msg in stSession.session_state.agent_messages:
    if msg.role != "system":
        with st.chat_message(msg.role):
            st.markdown(msg.content, unsafe_allow_html=True)

# -- handle pending tool call approval --
if stSession.session_state.pending_tool_execution:
    pending = stSession.session_state.pending_tool_execution
    with st.chat_message("assistant"):
        st.warning("**Command execution requested:**")
        for tc in pending["calls"]:
            args = parse_call_arguments(tc["arguments"])
            cwd = args.get("working_directory", "")
            st.code(args.get("command", ""), language="bash")
            if cwd:
                st.caption(f"Working directory: `{cwd}`")

        col1, col2 = st.columns(2)
        approved = col1.button("Approve", key="approve_exec")
        denied = col2.button("Deny", key="deny_exec")

        if approved or denied:
            tool_results = []
            for tc in pending["calls"]:
                args = parse_call_arguments(tc["arguments"])
                if approved:
                    cmd_output = run_command(
                        args["command"],
                        timeout=int(stSession.session_state.timeout),
                        cwd=args.get("working_directory"),
                    )
                    st.code(cmd_output, language="text")
                else:
                    cmd_output = "Command execution denied by user."
                tool_results.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc["call_id"],
                        "output": str(cmd_output),
                    }
                )

            stSession.session_state.pending_tool_execution = None

            with st.spinner("Continuing..."):
                continuation = chatAgent.create_turn(prompt=tool_results, stream=stream)

            if not stream:
                new_calls, cont_text, cont_tool_resp, cont_tool_outputs = (
                    _process_response_for_function_calls(continuation)
                )
                if cont_text:
                    st.markdown(cont_text)
                if new_calls:
                    stSession.session_state.pending_tool_execution = {
                        "calls": new_calls,
                    }
                    st.rerun()
                stSession.session_state.agent_messages.append(
                    AgentMessage(_role="assistant", _content=cont_text)
                )
            else:
                _, new_calls, cont_text, cont_callstack, cont_tool_outputs = (
                    _process_stream_for_function_calls(continuation)
                )
                if new_calls:
                    stSession.session_state.pending_tool_execution = {
                        "calls": new_calls,
                    }
                    st.rerun()
                stSession.session_state.agent_messages.append(
                    AgentMessage(_role="assistant", _content=cont_text)
                )

            try:
                stSession.save_chat_history(
                    stSession.session_state.latest_history_filename,
                    stSession.session_state.agent_messages,
                )
            except Exception as e:
                st.warning(f"Autosave failed: {e}")

prompt_raw = st.chat_input(
    placeholder="Say something...",
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
        prompt_response = ""
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
                        augmented_prompt = [
                            {"role": "user", "content": [txt_entity, img_entity]}
                        ]
                    else:
                        with st.spinner(f"Creating Docling Converter.... {f.name}"):
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

            message_placeholder = st.empty()
            if not stream:
                with st.spinner("Thinking...."):
                    response = chatAgent.create_turn(
                        prompt=augmented_prompt, stream=stream
                    )

                local_calls, prompt_response, tool_response, tool_outputs = (
                    _process_response_for_function_calls(response)
                )
                message_placeholder.markdown(prompt_response)

                with st.expander("Inference Stack"):
                    st.markdown(tool_response)
                    for output in tool_outputs:
                        with st.expander(f"Output: {output['label']}"):
                            st.code(output["content"])

                if local_calls:
                    stSession.session_state.pending_tool_execution = {
                        "calls": local_calls,
                    }
                    stSession.session_state.agent_messages.append(
                        AgentMessage(_role="assistant", _content=prompt_response)
                    )
                    st.rerun()
            else:
                final_response, local_calls, prompt_response, callstack_response, tool_outputs = (
                    _process_stream_for_function_calls(
                        chatAgent.create_turn(prompt=augmented_prompt, stream=stream)
                    )
                )

                with st.expander("Inference Stack"):
                    st.markdown(callstack_response)
                    for output in tool_outputs:
                        with st.expander(f"Output: {output['label']}"):
                            st.code(output["content"])

                if local_calls:
                    stSession.session_state.pending_tool_execution = {
                        "calls": local_calls,
                    }
                    stSession.session_state.agent_messages.append(
                        AgentMessage(_role="assistant", _content=prompt_response)
                    )
                    st.rerun()

        except Exception as e:
            st.error(f"Request failed: {e}")

        # add to history
        if prompt_response:
            stSession.session_state.agent_messages.append(
                AgentMessage(_role="assistant", _content=prompt_response)
            )

            # save latest messages in the last_chat json file on disk
            try:
                stSession.save_chat_history(
                    stSession.session_state.latest_history_filename,
                    stSession.session_state.agent_messages,
                )
            except Exception as e:
                st.warning(f"Autosave failed: {e}")
