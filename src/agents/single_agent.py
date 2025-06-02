"""
single_agent.py

This module defines the SingleAgent class, an implementation of a single agent
with system message loading capabilities for use in conversational AI applications.
It provides mechanisms to initialize an agent with a specific persona, model, and
optional document context, and to retrieve system messages based on these parameters.

Classes:
- SingleAgent: Implements a single agent with persona and document-based system message loading.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from streamlit.logger import get_logger
from typing_extensions import Annotated, TypedDict

from utils.cached_loader import load_prompt_files

from .agent import Agent

logger = get_logger(__name__)


class ChatState(TypedDict):
    """State definition for the chat workflow."""

    messages: Annotated[List[BaseMessage], add_messages]


class SingleAgent(Agent):
    """
    Implementation of a single agent with system message loading capabilities.
    """

    def __init__(
        self,
        agent_key: str,
        persona: str,
        model: Optional[str],
        temperature: Optional[float],
        documents: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Initialize a SingleAgent with a specific persona.

        Parameters:
        -----------
        agent_key : str
            Unique identifier for the agent.
        persona : str
            Path to the persona prompt file.
        model : str, optional
            The model to use for this agent. Defaults to "gpt-4o".
        documents : Optional[Union[str, List[str]]], optional
            Path(s) to document context file(s). Defaults to None.
        temperature : float, optional
            The temperature setting for response generation. Defaults to 0.7.
        """
        super().__init__(agent_key, model, temperature)
        self.persona = persona
        self.documents = documents

    def get_system_messages(self) -> List[Dict[str, str]]:
        """
        Get the system messages for this agent based on persona and documents.

        Returns:
        --------
        List[Dict[str, str]]
            A list of system messages for the agent.
        """
        logger.debug(
            "Loading system messages for SingleAgent %s with persona %s and documents %s",
            self.agent_key,
            self.persona,
            self.documents,
        )
        return load_prompt_files(self.persona, self.documents)

    def create_chain(self) -> Runnable:  # [dict[Any, Any], BaseMessage]:
        """
        Create and return a compiled state graph for this agent.

        Returns:
        --------
        CompiledStateGraph
            A compiled state graph representing the agent's workflow.
        """
        from agents import RESPONSE_TAG

        if self._chain is None:
            start_prompt = ChatPromptTemplate.from_messages(
                [MessagesPlaceholder("messages")]
            )
            # Single Agent is tagged as a response agent, this means
            # responses from this agent will be show in the UI
            llm = self._get_azure_chat_openai(tag=RESPONSE_TAG)
            chain = start_prompt | llm
            self._chain = chain
            return chain

        return self._chain
