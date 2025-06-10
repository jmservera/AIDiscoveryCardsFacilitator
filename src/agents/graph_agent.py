"""
graph_agent.py

This module defines the GraphAgent class, an implementation of a decision tree agent
with conditional routing capabilities using Azure AI Agents Service.
The GraphAgent creates a single Azure AI agent with complex instructions that include
the routing logic and knowledge about sub-agents.

Classes:
--------
GraphAgent : Agent
    Implements a decision-based agent using Azure AI Agents Service with routing instructions.

Dependencies:
-------------
- azure.ai.agents: For Azure AI Agents Service integration
"""

import os
from logging import getLogger
from typing import Any, Dict, List, Optional

from azure.ai.agents.models import Agent as AzureAgent

from .agent import Agent

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = getLogger(__name__)
logger.setLevel(LOGLEVEL)


class GraphAgent(Agent):
    """
    Implementation of a decision-based agent using Azure AI Agents Service.

    This agent creates a single Azure AI agent with complex instructions that include
    routing logic and knowledge about different scenarios. Instead of physically routing
    to different sub-agents, it responds as the appropriate persona based on the input.
    """

    def __init__(
        self,
        agent_key: str,
        condition: str,
        agents: List[Dict[str, str]],
        model: Optional[str],
        temperature: Optional[float] = 0.7,
    ) -> None:
        """
        Initialize a GraphAgent with conditional routing capabilities.

        Parameters:
        -----------
        agent_key : str
            Unique identifier for the agent.
        condition : str
            The condition prompt used to evaluate incoming messages and determine routing.
        agents : List[Dict[str, str]]
            List of agent configurations containing agent keys and their conditions.
        model : str, optional
            The model to use for this agent. Defaults to "gpt-4o".
        temperature : float, optional
            The temperature setting for response generation. Defaults to 0.7.
        """
        super().__init__(agent_key, model, temperature)
        self.condition = condition
        self.agents = agents

    def get_system_prompts(self) -> List[Dict[str, str]]:
        """
        Get the system messages for this graph agent.

        Returns:
        --------
        List[Dict[str, str]]
            An empty list as graph agents use complex instructions in the Azure AI agent.
        """
        return []

    def create_agent(self) -> AzureAgent:
        """
        Create and return an Azure AI agent for this graph agent with routing instructions.

        Returns:
        --------
        AzureAgent
            An Azure AI agent configured with conditional routing instructions.
        """
        client = self._get_agents_client()
        
        # Build complex instructions that include the routing logic
        instructions = f"""
You are a sophisticated assistant with conditional routing capabilities. 

ROUTING CONDITION:
{self.condition}

AVAILABLE AGENT TYPES:
"""
        
        # Add information about each available agent
        for agent_config in self.agents:
            agent_key = agent_config.get("agent", "unknown")
            agent_condition = agent_config.get("condition", "unknown condition")
            instructions += f"\n- {agent_key}: Used when {agent_condition}"
        
        instructions += f"""

INSTRUCTIONS:
1. Analyze the user's input carefully
2. Determine which agent type would be most appropriate based on the routing conditions above
3. Respond as that specific agent type would respond
4. Include the agent type decision in your thinking but focus your response on being helpful as that agent

Always be helpful and provide accurate, relevant responses based on the determined agent type.
"""
        
        # Create the Azure AI agent
        azure_agent = client.create_agent(
            model=self.model,
            name=f"{self.agent_key}_graph_agent",
            description=f"Graph agent with routing for {self.agent_key}",
            instructions=instructions.strip(),
            temperature=self.temperature
        )
        
        logger.info(
            "Created Azure AI agent %s for GraphAgent %s with %d routing options",
            azure_agent.id,
            self.agent_key,
            len(self.agents)
        )
        
        return azure_agent
