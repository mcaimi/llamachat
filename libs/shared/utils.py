#!/usr/bin/env python

# buil requests header object
def build_header(api_key: str) -> dict:
    if api_key is None or api_key == "":
        api_key = "apikey_openai"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"  # Bearer token format
    }
    return headers