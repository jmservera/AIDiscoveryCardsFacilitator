"""
Agent implementation with system message loading capabilities

This module extends the base Agent classes from agent_registry.py with
implementation details for loading system messages from prompt files.
"""

from typing import Dict, List

from streamlit.logger import get_logger

from agent_registry import MultiAgent, SingleAgent
from utils.openai_utils import load_prompt_files

logger = get_logger(__name__)


class SingleAgentImpl(SingleAgent):
    """
    Implementation of a single agent with system message loading capabilities.
    """

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


class MultiAgentImpl(MultiAgent):
    """
    Implementation of a multi-agent with system message loading capabilities.
    """

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
