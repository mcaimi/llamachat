# LlamaChat

[![Docker Repository on Quay](https://quay.io/repository/marcocaimi/llamachat/status "Docker Repository on Quay")](https://quay.io/repository/marcocaimi/llamachat)

**LlamaChat** is a web application that allows users to chat with Large Language Models (LLMs). It provides flexibility to interact with LLMs either locally (on-prem) using technologies like **vLLM** or **Ollama** or through remote **OpenAI-compatible endpoints**.

With LlamaChat, users can easily chat with powerful LLMs both in private environments (for enhanced privacy and control) and using cloud-based APIs for scalability.

## Features

- **Unified Chat**: Interact with LLMs hosted on-premises or remotely using **llama-stack**
- **User-Friendly Interface**: Simple web interface to initiate and maintain conversations with LLMs.
- **API Key Support**: Securely manage and switch between different API keys (for remote services).
- **Multiple Models**: Support for multiple LLM models such as GPT-based models, Ollama models, and others.
- **RAG Support**: Chat with your documents, using a vector database as embeddings backend
- **Safety Shields**: Add safety guardrails to user prompts
- **Agents Support**: Use tools to enhance the LLM capabilities

## Screenshot

![LlamaChat Screenshot](assets/screenshot.png)

## Installation

### Prerequisites

- Python 3.12+
- Streamlit
- An Ollama/vLLM instance or a public OpenAI-Compatible API endpoint.
- Llama-Stack instance running on-prem or in the cloud.


