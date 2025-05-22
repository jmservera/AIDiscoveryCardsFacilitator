"""
Agent registry for Discovery Cards Agent (unified YAML version)

Loads agent definitions from the unified pages.yaml and provides lookup utilities.
"""

from pathlib import Path

import yaml

PAGES_FILE = Path(__file__).parent / "pages.yaml"


class AgentRegistry:
    def __init__(self, pages_file=PAGES_FILE):
        with open(pages_file, encoding="utf-8") as f:
            self._agents = yaml.safe_load(f)["agents"]

    def get(self, agent_key):
        return self._agents.get(agent_key)

    def all(self):
        return self._agents


# Singleton instance
agent_registry = AgentRegistry()
