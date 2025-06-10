"""
single_agent.py

This module defines the SingleAgent class, an implementation of a single agent
with system message loading capabilities for use with Azure AI Agents Service.
It provides mechanisms to initialize an agent with a specific persona, model, and
optional document context, and to create Azure AI agents with appropriate instructions.

Classes:
--------
SingleAgent : Agent
    Implements a single agent with persona and document-based Azure AI agent creation.
"""

import os
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from azure.ai.agents.models import Agent as AzureAgent
from utils.cached_loader import load_prompt_files

from .agent import Agent

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = getLogger(__name__)
logger.setLevel(LOGLEVEL)


class SingleAgent(Agent):
    """
    Implementation of a single agent using Azure AI Agents Service with system message loading capabilities.
    """

    def __init__(
        self,
        agent_key: str,
        persona: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        documents: Optional[Union[str, frozenset[str]]] = None,
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

    def get_system_prompts(self) -> List[Dict[str, str]]:
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

    def create_agent(self) -> AzureAgent:
        """
        Create and return an Azure AI agent for this single agent.

        Returns:
        --------
        AzureAgent
            An Azure AI agent configured with the persona and document instructions.
        """
        client = self._get_agents_client()
        
        # Get system prompts and combine them into instructions
        system_prompts = self.get_system_prompts()
        instructions = ""
        
        for prompt in system_prompts:
            if prompt.get("role") == "system":
                instructions += prompt.get("content", "") + "\n\n"
        
        # Create the Azure AI agent
        azure_agent = client.create_agent(
            model=self.model,
            name=f"{self.agent_key}_agent",
            description=f"Single agent for {self.agent_key}",
            instructions=instructions.strip(),
            temperature=self.temperature
        )
        
        logger.info(
            "Created Azure AI agent %s for SingleAgent %s",
            azure_agent.id,
            self.agent_key
        )
        
        return azure_agent
