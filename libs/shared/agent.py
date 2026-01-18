#!/usr/bin/env python
#
# Responses API Wrapper Class
#

class AgentSession:
    def __init__(self, agent):
        self.agent = agent
        self.continuity_id = None

    def generate(self, prompt):
        api_response = self.agent.client.responses.create(
            model = self.agent.model,
            instructions = self.agent.instructions,
            tools = self.agent.tools,
            previous_response_id = self.continuity_id,
            input = prompt,
        )
        self.continuity_id = api_response.id
        return api_response

    def forget(self):
        # reset continuity id and start from scratch
        self.continuity_id = None

class Agent:
    def __init__(self, llamastack_client, model, instructions, tools = None):
        self.model = model
        self.instructions = instructions
        self.client = llamastack_client
        self.tools = tools

        # agent session object
        self.session = AgentSession(self)

    def create_turn(self, prompt):
        return self.session.generate(prompt)

    def reset_turn(self):
        self.session.forget()