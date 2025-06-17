#!/usr/local/bin python

import os

# settings class, wraps many configuration aspects of the LLM application
class Properties(object):
    def __init__(self, dotenv_value_dict: dict) -> None:
        try:
            # init session state tracker
            self.config_parameters = dict()

            # setup default initial variables, look at the env var file
            self.config_parameters["api_base_url"] = dotenv_value_dict.get("DEFAULT_LOCAL_API", "http://localhost:11434/api")
            self.config_parameters["default_local_api"] = dotenv_value_dict.get("DEFAULT_LOCAL_API", "http://localhost:11434/api")
            self.config_parameters["default_cloud_api"] = dotenv_value_dict.get("DEFAULT_CLOUD_API", "http://your_cloud_provider/api")
            self.config_parameters["available_models"] = ["llama3"]
            self.config_parameters["custom_endpoint"] = ""
            self.config_parameters["system_prompt"] = dotenv_value_dict.get("DEFAULT_SYS_PROMPT", "You are an helpful assistant.")
            self.config_parameters["messages"] = [{"role": "system", "content": self.config_parameters.get("system_prompt")}]
            self.config_parameters["history_dir"] = dotenv_value_dict.get("HISTORY_DIR", "history")
            self.config_parameters["latest_history_filename"] = dotenv_value_dict.get("LATEST_HISTORY_FILE", "latest_chat.json")
            self.config_parameters["api_key"] = dotenv_value_dict.get("API_KEY", "unused")

            # vector database settings
            self.config_parameters["chromadb_host"] = dotenv_value_dict.get("CHROMADB_HOST", "http://localhost:8000")
            self.config_parameters["chromadb_collection"] = dotenv_value_dict.get("CHROMADB_COLLECTION", "redhat")

            # feature switches
            self.config_parameters["enable_rag"] = False

            # syntactic sugar
            for k in self.config_parameters.keys():
                setattr(self, k, self.config_parameters.get(k))

            # perform sanity check
            self.bootup_check()
        except Exception as e:
            raise e

    # sanity check at bootup
    def bootup_check(self) -> None:
        os.makedirs(self.config_parameters.get("history_dir"), exist_ok=True)

    # session variables
    def get_properties_object(self) -> dict:
        return self.config_parameters
