#!/usr/bin/env python
#
# Functions that helps handling responses from AI agents
#

def dict_to_markdown_table(data):
    """
    Convert a dictionary of key-value pairs into a Markdown-formatted table.

    Args:
        data (dict): Dictionary with keys as column headers and values as rows.

    Returns:
        str: Markdown-formatted table string.
    """

    # Get the keys (column headers) from the input dictionary
    headers = list(data.keys())
    values = list(data.values())

    # Initialize the Markdown table string
    markdown_table = f"| {'|'.join(headers)} |\n"  # Header row
    markdown_table += f"| {'---|' * len(headers)} \n"  # Separator row

    # Iterate over each key-value pair in the dictionary
    row = f"| {'|'.join([str(v) for v in values])} |\n"
    markdown_table += row

    # done
    return markdown_table

def format_mcp_response(mcp_call):
    import json

    info = f"| Field | Value |\n| ---|--- |\n"
    info += f"| **Tool** | `{mcp_call.name}` |\n"
    info += f"| **Server** | `{mcp_call.server_label}` |\n"
    info += f"| **ID** | `{mcp_call.id}` |\n"
    info += f"| **Arguments** | `{mcp_call.arguments}` |\n"
    if mcp_call.error:
        info += f"| **Error** | `{mcp_call.error}` |\n"

    output_data = None
    if mcp_call.output:
        try:
            parsed = json.loads(mcp_call.output)
            output_data = {"label": mcp_call.name, "content": json.dumps(parsed, indent=2)}
        except json.JSONDecodeError:
            output_data = {"label": mcp_call.name, "content": mcp_call.output}

    return info, output_data

def format_mcp_list_tools(mcp_list_tools):
    """
    Print MCP list tools in a nicely formatted way.

    Args:
        mcp_list_tools (object): MCP server object containing server_label, id,
            and a list of tool objects.
    """

    # preformat call stack
    tool_list_response = dict_to_markdown_table(
        {
            "MCP Server": mcp_list_tools.server_label,
            "ID": mcp_list_tools.id,
            "Available Tools": len(mcp_list_tools.tools)
        }
    )
    
    # Iterate over each tool in the MCP server's list of tools
    for i, tool in enumerate(mcp_list_tools.tools, 1):
        tool_parameters: str = ""
        # Parse and display the input schema of the current tool
        if tool.input_schema:
            properties = tool.input_schema['properties']
            required = tool.input_schema.get('required', [])

            # iterate over parameters           
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'unknown')
                param_desc = param_info.get('description', 'No description')

                tool_parameters += f"     • {param_name} ({param_type})"
                if param_desc:
                    tool_parameters += f"       {param_desc}"
        
        # format tool list
        tool_list_response += dict_to_markdown_table(
            {
                f"Tool {i}": tool.name,
                f"Tool {i} Description": tool.description,
                f"Tool {i} Parameters": tool_parameters,
            }
        )
        
    # return list
    return tool_list_response

def format_response(response) -> (str, str, list):
    output_response = ""
    tool_call_response = ""
    tool_outputs = []

    tool_call_response = dict_to_markdown_table(
        {
            "ID": response.id,
            "Model": response.model,
            "Timestamp": response.created_at,
            "Status": response.status,
            "Input Tokens": f":violet-badge[{response.usage.input_tokens}]",
            "Output Tokens": f":orange-badge[{response.usage.output_tokens}]",
            "Total Tokens": f":grey-badge[{response.usage.total_tokens}]",
        }
    )

    for i, output_item in enumerate(response.output):
        match output_item.type:
            case "text" | "message":
                content = output_item.content[0]
                match content.type:
                    case "output_text":
                        output_response += f"{content.text}"
                    case "refusal":
                        output_response += f"{content.refusal}"
            case "file_search_call":
                tool_call_response += f"| Field | Value |\n| ---|--- |\n"
                tool_call_response += f"| **Type** | `file_search` |\n"
                tool_call_response += f"| **ID** | `{output_item.id}` |\n"
                tool_call_response += f"| **Status** | `{output_item.status}` |\n"
                tool_call_response += f"| **Queries** | `{', '.join(output_item.queries)}` |\n"
                if output_item.results:
                    tool_outputs.append({"label": f"file_search:{output_item.id}", "content": str(output_item.results)})
            case "mcp_call":
                info, output_data = format_mcp_response(output_item)
                tool_call_response += info
                if output_data:
                    tool_outputs.append(output_data)
            case _:
                output_response += f"Response content: {output_item.content}"

    return output_response, tool_call_response, tool_outputs


def format_streaming_response(response) -> (str, str, dict | None):
    streaming_text_fragment = ""
    response_callstack = ""
    tool_output = None

    match (response.type):
        case "response.output_text.delta":
            streaming_text_fragment = f"{response.delta}"
        case "response.output_item.done":
            item = response.item
            match item.type:
                case "mcp_call":
                    info, tool_output = format_mcp_response(item)
                    response_callstack = info
                case "file_search_call":
                    response_callstack += f"| Field | Value |\n| ---|--- |\n"
                    response_callstack += f"| **Type** | `file_search` |\n"
                    response_callstack += f"| **ID** | `{item.id}` |\n"
                    response_callstack += f"| **Status** | `{item.status}` |\n"
                    response_callstack += f"| **Queries** | `{', '.join(item.queries)}` |\n"
                    if item.results:
                        tool_output = {"label": f"file_search:{item.id}", "content": str(item.results)}
        case "response.in_progress"|"response.created":
            pass
        case "response.completed":
            response_callstack = dict_to_markdown_table(
                {
                    "ID": response.response.id,
                    "Model": response.response.model,
                    "Timestamp": response.response.created_at,
                    "Status": response.response.status,
                    "Input Tokens": f":violet-badge[{response.response.usage.input_tokens}]",
                    "Output Tokens": f":orange-badge[{response.response.usage.output_tokens}]",
                    "Total Tokens": f":grey-badge[{response.response.usage.total_tokens}]",
                }
            )
        case _:
            pass

    return streaming_text_fragment, response_callstack, tool_output