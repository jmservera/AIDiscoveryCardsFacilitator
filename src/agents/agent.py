"""
agent.py

This module defines the base Agent class for implementing conversational agents
that interact with Azure OpenAI services via LangGraph workflows. This replaces
direct Azure OpenAI API calls with LangGraph-based chat workflows while maintaining
backward compatibility.

MIGRATION NOTE: This module has been refactored to use LangGraph and LangChain's
AzureChatOpenAI instead of direct openai.AzureOpenAI client usage. The interface
remains the same for backward compatibility with existing code.

Classes:
---------
Agent
    Base class for agent implementations, now using LangGraph workflows for
    chat completion instead of direct Azure OpenAI API calls.

Dependencies:
-------------
- os
- typing (Dict, List)
- workflows.chat_graph (LangGraph implementation)
- streamlit
- streamlit.logger
"""

import abc
import os
from logging import getLogger
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import chainlit as cl
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import AzureChatOpenAI

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

logger = getLogger(__name__)


@cl.cache
def _create_llm(
    azure_endpoint: str,
    api_version: str,
    azure_deployment: Optional[str],
    temperature: Optional[float],
    tag: Optional[str],
) -> AzureChatOpenAI:
    # Use Azure identity for authentication
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    logger.info(
        "Creating AzureChatOpenAI instance with endpoint: %s, deployment: %s",
        azure_endpoint,
        azure_deployment,
    )
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
        azure_deployment=azure_deployment,
        temperature=temperature,
        streaming=True,
        stream_usage=True,
        tags=[tag] if tag else None,
    )


class Agent(abc.ABC):
    """
    Base class for agent implementations using LangGraph workflows.

    MIGRATION NOTE: This class now uses LangGraph and LangChain's AzureChatOpenAI
    for interacting with Azure OpenAI, replacing direct openai.AzureOpenAI client usage.
    The interface remains the same for backward compatibility.

    Attributes:
    -----------
    agent_key : str
        Unique identifier for the agent.
    model : str
        The model to use for this agent. Defaults to "gpt-4o".
    temperature : float
        The temperature setting for response generation.
    """

    def __init__(
        self, agent_key: str, model: Optional[str], temperature: Optional[float]
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
        self._llm: Optional[AzureChatOpenAI] = None
        self._chain: Optional[Runnable] = None

    def _get_azure_chat_openai(self, tag: Optional[str] = None) -> AzureChatOpenAI:
        """
        Create and return an AzureChatOpenAI instance with authentication.

        Returns:
        --------
        AzureChatOpenAI
            Configured Azure OpenAI chat model instance.
        """

        if self._llm is None:
            self._llm = _create_llm(
                AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_VERSION,
                self.model,
                self.temperature,
                tag,
            )

        return self._llm

    @abc.abstractmethod
    def create_chain(self) -> Runnable:
        """
        Create and return a compiled state graph for this agent.

        Returns:
        --------
        RunnableSerializable
            An invocable chain.
        """
        pass

    def _convert_to_langchain_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """
        Convert message dictionaries to LangChain message format.

        Parameters:
        -----------
        messages : List[Dict[str, str]]
            List of message dictionaries with 'role' and 'content' keys.

        Returns:
        --------
        List[BaseMessage]
            List of LangChain message objects.
        """
        langchain_messages = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                logger.warning(f"Unknown message role: {role}")
                # Default to human message for unknown roles
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    def astream(
        self, messages: List[Dict[str, str]], config: RunnableConfig
    ) -> AsyncIterator[Any]:
        try:

            # langchain_messages = self._convert_to_langchain_messages(messages)
            chain = self.create_chain()
            full_messages = self.get_system_prompts() + messages

            return chain.astream(
                {"messages": full_messages}, config=config, stream_mode="messages"
            )
        except Exception as e:
            logger.exception(
                "Async LangGraph execution failed, using fallback response: %s", e
            )
            raise e

    @abc.abstractmethod
    def get_system_prompts(self) -> List[Dict[str, str]]:
        """
        Get the system messages for this agent.

        Returns:
        --------
        List[Dict[str, str]]
            A list of system messages for the agent.
        """
        pass
