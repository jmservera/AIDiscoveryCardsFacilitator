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

import os
from typing import Any, Dict, Iterable, List

import streamlit as st
from dotenv import load_dotenv
from streamlit.logger import get_logger

# Import LangGraph-based chat implementation
from workflows.chat_graph import ChatGraph

load_dotenv()

logger = get_logger(__name__)


class Agent:
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
        # Initialize LangGraph chat workflow
        self._chat_graph = ChatGraph(model=self.model, temperature=self.temperature)

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
    def format_messages(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Format messages to be compatible with the chat workflow.

        MIGRATION NOTE: This method now returns a list of dictionaries instead of 
        ChatCompletionMessageParam to work with LangGraph workflows.

        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys

        Returns:
            A list of properly formatted messages compatible with LangGraph workflows
        """
        # Ensure the messages have the correct structure for LangGraph
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def create_chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        """
        Create and return a new chat completion request using LangGraph workflow.

        MIGRATION NOTE: This method now uses LangGraph workflows instead of 
        direct Azure OpenAI API calls, but maintains the same interface for
        backward compatibility.

        Parameters:
        -----------
        messages : List[Dict[str, str]]
            List of message objects with role and content.

        Returns:
        --------
        Any
            A streaming response compatible with the original OpenAI format.
        """
        logger.debug(
            "Creating LangGraph chat completion for %d messages with model %s",
            len(messages),
            self.model,
        )
        
        # Format messages for LangGraph workflow
        formatted_messages = self.format_messages(messages)
        
        # Use LangGraph workflow for chat completion
        return self._chat_graph.create_chat_completion_sync(formatted_messages)
