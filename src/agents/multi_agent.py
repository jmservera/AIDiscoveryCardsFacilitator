"""
multi_agent.py

This module defines the MultiAgent class, an extension of the Agent class, designed to
manage multiple personas and optionally associate them with document contexts.
The MultiAgent class provides functionality to load system messages for each persona,
with or without associated documents, leveraging utility functions for prompt file loading.

Classes:
- MultiAgent: Represents an agent capable of handling multiple personas and
  loading their corresponding system messages, optionally paired with document contexts.

Dependencies:
-------------
- typing: For type annotations.
- streamlit.logger: For logging.
- utils.openai_utils.load_prompt_files: For loading prompt files.
- .agent.Agent: Base Agent class.

"""

from typing import Dict, List, Optional, Union

from streamlit.logger import get_logger

from utils.openai_utils import load_prompt_files

from .agent import Agent

logger = get_logger(__name__)


class MultiAgent(Agent):
    """
    Implementation of a multi-agent with system message loading capabilities.
    """

    def __init__(
        self,
        agent_key: str,
        personas: List[str],
        model: str = "gpt-4o",
        documents: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.7,
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
        self.personas = personas
        self.documents = documents

    def get_system_messages(self) -> List[Dict[str, str]]:
        """
        Get the system messages for this agent based on personas and documents.

        Returns:
        --------
        List[Dict[str, str]]
            A list of system messages for the agent.
        """
        logger.debug(
            "Loading system messages for MultiAgent %s with personas %s and documents %s",
            self.agent_key,
            self.personas,
            self.documents,
        )

        messages = []

        # If documents are provided, pair each persona with a document
        # Otherwise, just use the personas
        if self.documents:
            for i, persona in enumerate(self.personas):
                persona_docs = None
                if i < len(self.documents):
                    persona_docs = self.documents[i]

                persona_messages = load_prompt_files(persona, persona_docs)
                messages.extend(persona_messages)
        else:
            # No documents, just load personas
            for persona in self.personas:
                persona_messages = load_prompt_files(persona)
                messages.extend(persona_messages)

        return messages
