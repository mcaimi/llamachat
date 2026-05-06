#/usr/bin/env python
#
# Wrapper for APIs in OGX
#
#   eg: http://ogx-endpoint:port/v1beta/connections
#

try:
    from pprint import pprint
    import requests
except ImportError as e:
    print(f"Error importing libraries: {e}")


# wrapper for the Entity object
class Entity(object):
    """
        Example: for connections APIs:

        Wrap and maintain a description of a connector
        registered into the ogx config definition file.

        apis:
        - connectors

        connectors:
            - connector_id: "mcp::testconnection"
              connector_type: mcp
              url: "http:///localhost:8000/sse"
              server_label: "MCP Server"
              server_name: "server name",
              server_description: "description",
              server_version: "version"

        providers:
            tool_runtime:
                - provider_id: model-context-protocol
                  provider_type: inline::model-context-protocol
                  config: {}
    """
    def __init__(self, item: dict):
        if not isinstance(item, dict):
            raise RuntimeError("Wrong type, item must be dict")

        # parse dictionary
        self.item = item
        for key in self.item.keys():
            setattr(self, key, self.item.get(key))


# Wrapper api manager
class WrapperAPI(object):
    """
        Manages connections to API endpoints
    """
    def __init__(self, url, remote_endpoint: str, verify: bool = False):
        self.verify = False
        self.ogxUrl = url
        self.ogxEndpoint = remote_endpoint
        self.endpoint = f"{self.ogxUrl}/{self.ogxEndpoint}"

    # connections list
    def list(self) -> list:
        try:
            response = requests.get(self.endpoint, verify=self.verify)
        except Exception as e:
            raise e

        print(response)
        # parse response
        if response.status_code == 200:
            rJson = response.json()
            return [Entity(item) for item in rJson.get("data")]
        else:
            return []

# Connections API Wrapper
class ConnectorsAPI(WrapperAPI):
    def __init__(self, url: str, verify: bool = False):
        super().__init__(url=url, remote_endpoint="/v1beta/connectors", verify=verify)

# Tools API Wrapper
class ToolsAPI(WrapperAPI):
    def __init__(self, url: str, verify: bool = False):
        super().__init__(url=url, remote_endpoint="/v1/tools", verify=verify)