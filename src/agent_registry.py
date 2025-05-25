"""
Agent registry for Discovery Cards Agent (unified YAML version)

Loads agent definitions from the unified pages.yaml and provides lookup utilities.

Classes:
---------
Agent:
    Base class for agent implementations.

SingleAgent:
    Implementation of a single agent with one persona.

MultiAgent:
    Implementation of multiple agents with multiple personas.

AgentRegistry:
    A class to manage and retrieve agent definitions from the unified YAML file.

Singletons:
-----------
agent_registry:
    A singleton instance of the AgentRegistry class for global use.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from streamlit.logger import get_logger

PAGES_FILE = Path(__file__).parent / "config/pages.yaml"

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
        self, agent_key: str, model: str = "gpt-4o", temperature: float = 0.7
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
            The temperature setting for response generation. Defaults to 0.7.
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


class SingleAgent(Agent):
    """
    Implementation of a single agent with one persona.

    Attributes:
    -----------
    persona : str
        Path to the persona prompt file.
    documents : Optional[Union[str, List[str]]]
        Path(s) to document context file(s).
    """

    def __init__(
        self,
        agent_key: str,
        persona: str,
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
        # This will be implemented using the load_prompt_files function
        # For now, we're just defining the interface
        pass


class MultiAgent(Agent):
    """
    Implementation of multiple agents with multiple personas.

    Attributes:
    -----------
    personas : List[str]
        List of paths to the persona prompt files.
    documents : Optional[List[str]]
        List of paths to document context files.
    """

    def __init__(
        self,
        agent_key: str,
        personas: List[str],
        model: str = "gpt-4o",
        documents: Optional[List[str]] = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize a MultiAgent with multiple personas.

        Parameters:
        -----------
        agent_key : str
            Unique identifier for the agent.
        personas : List[str]
            List of paths to the persona prompt files.
        model : str, optional
            The model to use for this agent. Defaults to "gpt-4o".
        documents : Optional[List[str]], optional
            List of paths to document context files. Defaults to None.
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
        # This will be implemented using the load_prompt_files function
        # For now, we're just defining the interface
        pass


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
        # Import here to avoid circular imports
        from agent_impl import MultiAgentImpl, SingleAgentImpl

        agent_config = self.get(agent_key)
        if not agent_config:
            logger.warning("Agent key '%s' not found in registry", agent_key)
            return None

        try:
            model = agent_config.get("model", "gpt-4o")
            temperature = agent_config.get("temperature", 0.7)

            # Determine if it's a single or multi agent based on the config
            if "persona" in agent_config:
                # Single agent
                documents = None
                if "document" in agent_config:
                    documents = agent_config["document"]
                elif "documents" in agent_config:
                    documents = agent_config["documents"]

                logger.info(
                    "Creating SingleAgent for key '%s' with model '%s'",
                    agent_key,
                    model,
                )
                return SingleAgentImpl(
                    agent_key=agent_key,
                    persona=agent_config["persona"],
                    model=model,
                    documents=documents,
                    temperature=temperature,
                )
            elif "personas" in agent_config:
                # Multi agent
                logger.info(
                    "Creating MultiAgent for key '%s' with model '%s'", agent_key, model
                )
                return MultiAgentImpl(
                    agent_key=agent_key,
                    personas=agent_config["personas"],
                    model=model,
                    documents=agent_config.get("documents"),
                    temperature=temperature,
                )
            else:
                logger.error(
                    "Invalid agent configuration for key '%s': missing 'persona' or 'personas'",
                    agent_key,
                )
                raise ValueError(
                    f"Invalid agent configuration for '{agent_key}': missing 'persona' or 'personas'"
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
