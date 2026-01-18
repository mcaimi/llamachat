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
    """
    Print MCP call in a nicely formatted way.

    Args:
        mcp_call (object): MCP call object containing name, server_label, id, arguments,
            error, and output.
    """

    # preformat mcp call stack
    mcp_call_response = dict_to_markdown_table(
        {
            "name": mcp_call.name,
            "id": mcp_call.id,
            "arguments": mcp_call.arguments,
            "server_label": mcp_call.server_label,
            "error": mcp_call.error,
        }
    )
    
    # Examine output
    if mcp_call.output:
        try:
            import json

            # Attempt to parse the JSON output of the MCP tool call
            parsed_output = json.loads(mcp_call.output)

            # Pretty-print the parsed JSON output
            mcp_call_response += dict_to_markdown_table(
                {
                    "output": parsed_output
                }
            )
        except json.JSONDecodeError:
            # If not valid JSON, print the raw output as-is
            print(f"   {mcp_call.output}")

    # return mcp call stack
    return mcp_call_response

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

def format_response(response) -> (str, str):
    """
    Function to format the response from the OpenAI API.

    Returns:
        tuple: A tuple containing two strings, the formatted output and the tool call response.
    """

    # Initialize variables to store the formatted output and tool call response
    output_response = ""
    tool_call_response = ""

    # Prepare the tool call response string with relevant information
    # Use f-string formatting for cleaner code
    tool_call_response = dict_to_markdown_table(
        {
            "ID": response.id,
            "Model": response.model,
            "Timestamp": response.created_at,
            "Status": response.status,
        }
    )
    
    # Iterate over each output item in the response
    for i, output_item in enumerate(response.output):              
        if output_item.type in ("text", "message"):
            # Append text content to the output_response string
            output_response += f"{output_item.content[0].text}"
        elif output_item.type == "file_search_call":
            # Extract relevant information from the file search call
            tool_call_response += f"### Msg Type: {output_item.type} - Tool Call ID: {output_item.id}, Tool Status: {output_item.status}\n"
            tool_call_response += f"### Queries: {', '.join(output_item.queries)}\n"
            # Append results to the output_response string if available
            tool_call_response += f"###  Results: {output_item.results if output_item.results else 'None'}"
        elif output_item.type == "mcp_list_tools":
            # Call function to print MCP list tools (not shown in this code snippet)
            tool_call_response += format_mcp_list_tools(output_item)
        elif output_item.type == "mcp_call":
            # Call function to print MCP call (not shown in this code snippet)
            tool_call_response += format_mcp_response(output_item)
        else:
            # Append generic response content to the output_response string
            output_response += f"Response content: {output_item.content}"

    return output_response, tool_call_response