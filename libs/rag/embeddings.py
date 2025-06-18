#!/usr/bin/env/python

from typing import Callable
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding


def openai_instance(base_url="http://localhost:11434", model="llama2:7b", api_key=None) -> Callable:
    return OpenAILikeEmbedding(api_base=base_url, model_name=model, api_key=api_key)


def ollama_instance(base_url="http://localhost:11434", model="llama2:7b", api_key=None) -> Callable:
    return OllamaEmbedding(api_base=base_url, model_name=model, api_key=api_key)