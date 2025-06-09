"""
Agent registry for Discovery Cards Agent (unified YAML version)

Loads agent definitions from the unified pages.yaml and provides lookup utilities.

Classes:
---------
Agent:
    Base class for agent implementations.

SingleAgent:
    Implementation of a single agent with one persona.

AgentRegistry:
    A class to manage and retrieve agent definitions from the unified YAML file.

Singletons:
-----------
agent_registry:
    A singleton instance of the AgentRegistry class for global use.
"""

import os
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional

import yaml

from .agent import Agent
from .graph_agent import GraphAgent
from .single_agent import SingleAgent
from .supervisor_agent import SupervisorAgent
from .react_agent import ReactAgent

PAGES_FILE = Path(__file__).parent.parent / "config/pages.yaml"


LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = getLogger(__name__)
logger.setLevel(LOGLEVEL)


class AgentRegistry:
    """
    A registry for managing agent definitions loaded from a unified YAML file.

    Methods:
    --------
    __init__(pages_file):
        Initializes the registry by loading agent definitions from the specified YAML file.

    get(agent_key):
        Retrieves the definition of a specific agent by its key.

    get_agent(agent_key):
        Creates and returns an Agent instance for the specified agent key.

    all():
        Returns all agent definitions as a dictionary.
    """

    def __init__(self, pages_file: Path = PAGES_FILE) -> None:
        """
        Initialize the AgentRegistry by loading agent definitions from the YAML file.

        Parameters:
        -----------
        pages_file : Path, optional
            The path to the YAML file containing agent definitions. Defaults to PAGES_FILE.
        """
        logger.info("Loading agent definitions from %s", pages_file)
        with open(pages_file, encoding="utf-8") as f:
            self._agents = yaml.safe_load(f)["agents"]

    def get(self, agent_key: str) -> Optional[Dict]:
        """
        Retrieve the definition of a specific agent by its key.

        Parameters:
        -----------
        agent_key : str
            The key of the agent to retrieve.

        Returns:
        --------
        dict or None
            The agent definition if found, otherwise None.
        """
        return self._agents.get(agent_key)

    def get_agent(self, agent_key: str) -> Optional[Agent]:
        """
        Create and return an Agent instance for the specified agent key.

        Parameters:
        -----------
        agent_key : str
            The key of the agent to create.

        Returns:
        --------
        Agent or None
            An instance of the appropriate Agent subclass if the key is found,
            otherwise None.

        Raises:
        -------
        ValueError:
            If the agent configuration is invalid or missing required fields.
        """

        agent_config = self.get(agent_key)
        if not agent_config:
            logger.warning("Agent key '%s' not found in registry", agent_key)
            return None

        try:
            model = agent_config.get("model", "gpt-4o")
            temperature = agent_config.get("temperature")

            # Determine if it's a single or multi agent based on the config
            if "persona" in agent_config:
                # Single agent
                documents = None
                if "document" in agent_config:
                    documents = agent_config["document"]
                elif "documents" in agent_config:
                    documents = frozenset(agent_config["documents"])

                logger.info(
                    "Creating SingleAgent for key '%s' with model '%s'",
                    agent_key,
                    model,
                )
                return SingleAgent(
                    agent_key=agent_key,
                    persona=agent_config["persona"],
                    model=model,
                    documents=documents,
                    temperature=temperature,
                )
            elif "condition" in agent_config:
                return GraphAgent(
                    agent_key=agent_key,
                    condition=agent_config["condition"],
                    model=model,
                    temperature=temperature,
                    agents=agent_config.get("agents", []),
                )
            elif "workers" in agent_config and "delegation_prompt" in agent_config:
                # Supervisor agent
                logger.info(
                    "Creating SupervisorAgent for key '%s' with model '%s'",
                    agent_key,
                    model,
                )
                return SupervisorAgent(
                    agent_key=agent_key,
                    workers=agent_config["workers"],
                    delegation_prompt=agent_config["delegation_prompt"],
                    model=model,
                    temperature=temperature,
                )
            elif "react_persona" in agent_config:
                # ReAct agent
                logger.info(
                    "Creating ReactAgent for key '%s' with model '%s'",
                    agent_key,
                    model,
                )
                return ReactAgent(
                    agent_key=agent_key,
                    persona=agent_config["react_persona"],
                    model=model,
                    temperature=temperature,
                    max_iterations=agent_config.get("max_iterations", 3),
                )
            else:
                logger.error(
                    "Invalid agent configuration for key '%s': missing required configuration fields",
                    agent_key,
                )
                raise ValueError(
                    f"Invalid agent configuration for '{agent_key}': missing required configuration fields"
                )
        except Exception as e:
            logger.exception("Error creating agent for key '%s': %s", agent_key, e)
            raise

        return None

    def all(self) -> Dict:
        """
        Retrieve all agent definitions.

        Returns:
        --------
        dict
            A dictionary of all agent definitions.
        """
        return self._agents


# Singleton instance
agent_registry = AgentRegistry()
