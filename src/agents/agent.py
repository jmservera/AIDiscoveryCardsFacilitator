"""
agent.py

This module defines the base Agent class for implementing conversational agents
that interact with Azure OpenAI services. It provides configuration for model
selection, temperature, and system messages, and includes methods for
initializing an authenticated Azure OpenAI client and creating chat completion
requests.

Classes:
---------
Agent
    Base class for agent implementations, providing methods for system message
    retrieval and chat completion creation using Azure OpenAI.

Dependencies:
-------------
- os
- typing (Dict, List)
- openai
- streamlit
- azure.identity
- streamlit.logger
"""

import os
from typing import Any, Dict, Iterable, List

import openai
import streamlit as st
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from streamlit.logger import get_logger

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

logger = get_logger(__name__)


class Agent:
    """
    Base class for agent implementations.

    Attributes:
    -----------
    agent_key : str
        Unique identifier for the agent.
    model : str
        The model to use for this agent.
    temperature : float
        The temperature setting for response generation.
    """

    def __init__(
        self, agent_key: str, model: str = "gpt-4o", temperature: float = 1
    ) -> None:
        """
        Initialize an Agent with configurable settings.

        Parameters:
        -----------
        agent_key : str
            Unique identifier for the agent.
        model : str, optional
            The model to use for this agent. Defaults to "gpt-4o".
        temperature : float, optional
            The temperature setting for response generation. Defaults to 1.
        """
        self.agent_key = agent_key
        self.model = model
        self.temperature = temperature

    def get_system_messages(self) -> List[Dict[str, str]]:
        """
        Get the system messages for this agent.

        Returns:
        --------
        List[Dict[str, str]]
            A list of system messages for the agent.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @st.cache_resource
    def get_client() -> openai.AzureOpenAI:
        """
        Get the Azure OpenAI client using DefaultAzureCredential.

        Returns:
        --------
        openai.AzureOpenAI
            An authenticated Azure OpenAI client.
        """
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

        return openai.AzureOpenAI(
            azure_ad_token_provider=token_provider,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

    @staticmethod
    def format_messages(
        messages: List[Dict[str, Any]],
    ) -> Iterable[ChatCompletionMessageParam]:
        """
        Format messages to be compatible with OpenAI's ChatCompletionMessageParam type.

        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys

        Returns:
            A list of properly formatted messages compatible with OpenAI's API
        """
        # The typing is handled by the ChatCompletionMessageParam, which will validate
        # that the dictionaries have the correct structure
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def create_chat_completion(self, messages: List[Dict[str, str]]) -> openai.Stream:
        """
        Create and return a new chat completion request.

        Parameters:
        -----------
        messages : List[ChatCompletionMessageParam]
            List of message objects with role and content.

        Returns:
        --------
        openai.Stream
            A streaming response from Azure OpenAI.
        """
        client = self.get_client()

        logger.debug(
            "Creating chat completion for %d messages with model %s",
            len(messages),
            self.model,
        )
        # Create and return a new chat completion request
        return client.chat.completions.create(
            model=self.model,
            messages=self.format_messages(messages),
            stream=True,
            stream_options={"include_usage": True},
            temperature=self.temperature,
        )
