#!/usr/local/bin python

import os
from yaml import safe_load, YAMLError
from libs.utils.parameters import Parameters
from libs.utils.console_utils import ANSIColors

# settings class, wraps many configuration aspects of the LLM application
class Properties(object):
    def __init__(self, config_file: str) -> None:
        self.config_file_name = config_file
        self.ttyWriter = ANSIColors()
        try:
            # init session state tracker
            self.config_parameters: Parameters = None
            try:
                self.load_config_parms()
            except Exception as e:
                raise e

            # perform sanity check
            self.bootup_check()
        except Exception as e:
            raise e

    def load_config_parms(self) -> None:
        try:
            with open(self.config_file_name, "r") as f:
                config_parms = safe_load(f)

            self.config_parameters = Parameters(config_parms)
        except YAMLError as e:
            self.ttyWriter.print_error(text=e)
            raise e
        except Exception as e:
            self.ttyWriter.print_error(text=e)
            raise e

    # sanity check at bootup
    def bootup_check(self) -> None:
        os.makedirs(self.config_parameters.openai.history_dir, exist_ok=True)

    # session variables
    def get_properties_object(self) -> dict:
        return self.config_parameters
