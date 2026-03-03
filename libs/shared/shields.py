#!/usr/bin/env python
#
#   Shields and Moderations API Wrapper
#

class ShieldOutput:
    def __init__(self, responseObject):
        self.results = responseObject
        self.outcome = []
        self.msg = ""

        self._parse()

    def _parse(self):
        # parse shield output
        if self.results is None:
            self.outcome.append(
                {
                    "flagged": False,
                }
            )
            
        for i, response_id in enumerate(self.results):
            for k, output_item in enumerate(response_id.results):
                match output_item.flagged:
                    case True:
                        self.outcome.append(
                            {
                                "flagged": True,
                                "user_message": output_item.user_message,
                                "violation_type": output_item.metadata.get("violation_type")
                            }
                        )
                    case _:
                        self.outcome.append(
                            {
                                "flagged": False,
                            }
                        )

    def flagged(self):
        return any(map(lambda x: x.get("flagged"), self.outcome))

    def str(self):
        for i in self.outcome:
            self.msg += f"Flagged: {i.get("flagged")} - {i.get("user_message")} - {i.get('violation_type')}"
        return self.msg

class ShieldEvaluation:
    def __init__(self, shield, llamastackClient):
        self.shield = shield
        self.client = llamastackClient

    def evaluate(self, prompt):
        shield_response = self.client.moderations.create(
            input = prompt,
            model = self.shield
        )

        return shield_response

class Shields:
    def __init__(self, shields, client):
        if type(shields) == list:
            self.shields = shields
            self.client = client
        else:
            raise Exception("Parameter 'shields' must be a list of shield models")
        
    def run_shields(self, prompt):
        # run shields
        evaluations = []
        for s in self.shields:
            # evaluate shields
            evaluations.append(ShieldEvaluation(s, self.client).evaluate(prompt=prompt))
        return evaluations or None

