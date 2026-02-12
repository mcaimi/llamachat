#!/usr/bin/env python
#
# Responses API Wrapper Class
#

# import shields
from .shields import ShieldEvaluation, Shields, ShieldOutput

class AgentSession:
    def __init__(self, agent):
        self.agent = agent
        self.continuity_id = None

    # perform inference, optionally with shields
    def generate(self, prompt):
        if self.agent.sampling_params is None:
            sampling_params = {}
        else:
            sampling_params = self.agent.sampling_params

        api_response = self.agent.client.responses.create(
            model = self.agent.model,
            instructions = self.agent.instructions,
            tools = self.agent.tools,
            previous_response_id = self.continuity_id,
            input = prompt,
            include = [
                    "web_search_call.action.sources",
                    "code_interpreter_call.outputs",
                    "computer_call_output.output.image_url",
                    "file_search_call.results",
                    "message.input_image.image_url",
                    "message.output_text.logprobs",
                    "reasoning.encrypted_content",
                ],
            stream = False,
            **sampling_params,
        )

        self.continuity_id = api_response.id
        return api_response

    def forget(self):
        # reset continuity id and start from scratch
        self.continuity_id = None

class Agent:
    def __init__(self,
                llamastack_client,
                model,
                instructions,
                tools = None,
                input_shields = None,
                output_shields = None,
                sampling_params: dict = None):
        self.model = model
        self.instructions = instructions
        self.client = llamastack_client
        self.tools = tools
        self.input_shields = input_shields
        self.output_shields = output_shields
        self.sampling_params = sampling_params

        # agent session object
        self.session = AgentSession(self)

    def _run_shield(self, shield, prompt):
        # evaluate input shields
        if shield:
            shields_evals = Shields(shield, self.client).run_shields(prompt)

            output = ShieldOutput(shields_evals)
            
            if output.flagged():
                return True, output
            else:
                return False, None
        else:
            return False, None

    def input_shield(self, prompt):
        return self._run_shield(self.input_shields, prompt)

    def output_shield(self, prompt):
        return self._run_shield(self.output_shields, prompt)

    def create_turn(self, prompt):
        return self.session.generate(prompt)

    def reset_turn(self):
        self.session.forget()