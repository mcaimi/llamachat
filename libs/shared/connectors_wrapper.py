#/usr/bin/env python
#
# Wrapper for the connections API in LlamaStack
#
#   http://llamastack-endpoint:port/v1beta/connections
#

try:
    from pprint import pprint
    import requests
except ImportError as e:
    print(f"Error importing libraries: {e}")


# wrapper for the ConnectorEntity object
class ConnectorEntity(object):
    """
        Wrap and maintain a description of a connector
        registered into the llamastack config definition file.

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
    def __init__(self, connectorItem: dict):
        if not isinstance(connectorItem, dict):
            raise RuntimeError("Wrong type, connectorItem must be dict")

        # parse dictionary
        self.connectorItem = connectorItem
        for key in self.connectorItem.keys():
            setattr(self, key, self.connectorItem.get(key))


# connections api manager
class ConnectorsAPI(object):
    """
        Manages connections to the connectors endpoint
        /v1beta/connectors
    """
    def __init__(self, url, endpoint: str = "/v1beta/connectors", verify: bool = False):
        self.verify = False
        self.llamaUrl = url
        self.llamaEndpoint = endpoint
        self.connectorsEndpoint = f"{self.llamaUrl}/{self.llamaEndpoint}"

    # connections list
    def list(self) -> list:
        try:
            response = requests.get(self.connectorsEndpoint, verify=self.verify)
        except Exception as e:
            raise e

        print(response)
        # parse response
        if response.status_code == 200:
            rJson = response.json()
            return [ConnectorEntity(item) for item in rJson.get("data")]
        else:
            return []
