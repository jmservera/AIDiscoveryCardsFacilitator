"""
Agent registry for Discovery Cards Agent (unified YAML version)

Loads agent definitions from the unified pages.yaml and provides lookup utilities.

Classes:
---------
AgentRegistry:
    A class to manage and retrieve agent definitions from the unified YAML file.

Singletons:
-----------
agent_registry:
    A singleton instance of the AgentRegistry class for global use.
"""

from pathlib import Path
from typing import Dict, Optional

import yaml
from streamlit.logger import get_logger

PAGES_FILE = Path(__file__).parent / "config/pages.yaml"

logger = get_logger(__name__)


class AgentRegistry:
    """
    A registry for managing agent definitions loaded from a unified YAML file.

    Methods:
    --------
    __init__(pages_file):
        Initializes the registry by loading agent definitions from the specified YAML file.

    get(agent_key):
        Retrieves the definition of a specific agent by its key.

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
