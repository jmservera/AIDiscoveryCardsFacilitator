"""
graph_agent.py

This module defines the GraphAgent class, an implementation of a decission tree agent
with system message loading capabilities for use in conversational AI applications.

Classes:
- GraphAgent: Implements a graph agent with persona and document-based system message loading.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from streamlit.logger import get_logger

from utils.openai_utils import load_prompt_files

from .agent import Agent

logger = get_logger(__name__)


class GraphAgent(Agent):
    """
    Implementation of a single agent with system message loading capabilities.
    """

    def __init__(
        self,
        agent_key: str,
        condition: str,
        agents: List[str],
        model: Optional[str],
        temperature: Optional[float] = 0.7,
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
        self.condition = condition
        self.agents = agents

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
