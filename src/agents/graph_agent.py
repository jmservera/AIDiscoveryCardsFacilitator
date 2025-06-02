"""
graph_agent.py

This module defines the GraphAgent class, an implementation of a decission tree agent
with system message loading capabilities for use in conversational AI applications.

Classes:
- GraphAgent: Implements a graph agent with persona and document-based system message loading.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from streamlit.logger import get_logger
from typing_extensions import Annotated

from utils.streamlit_context import with_streamlit_context

from .agent import Agent

logger = get_logger(__name__)


class AgentState(TypedDict):
    input: str
    output: str
    decision: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


class GraphAgent(Agent):
    """
    Implementation of a single agent with system message loading capabilities.
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

        return []

    @with_streamlit_context  # as it potentially uses streamlit caching we need to ensure the context is set
    def _agent_node(self, state):
        from .agent_registry import agent_registry

        agent = agent_registry.get_agent(state["decision"])
        if agent is None:
            logger.error(f"Agent {state['decision']} not found in registry.")
            raise ValueError(f"Agent {state['decision']} not found in registry.")

        messages = agent.get_system_messages()
        messages.append(state["messages"][-1])
        chain = agent.create_chain()
        msg = chain.invoke({"messages": messages})
        return {"output": msg}

    @with_streamlit_context  # as it potentially uses streamlit caching we need to ensure the context is set
    def _start_agent(self, state) -> Dict[str, Any]:
        """
        Start the agent based on the condition evaluation.

        Parameters:
        -----------
        state : ChatState
            Current state containing the conversation messages.

        Returns:
        --------
        Dict[str, Any]
            Updated state with the selected agent's response.
        """
        # Here you would implement the logic to select and start the appropriate agent
        # based on the condition. This is a placeholder implementation.
        start_prompt = ChatPromptTemplate.from_messages([("system", self.condition)])
        chain = start_prompt | self._get_azure_chat_openai()
        input = state["input"] if "input" in state else state["messages"][-1].content

        response = chain.invoke({"input": input})
        # take the decision from the response
        decision = response.content.strip().lower()
        # Populate the messages and output fields
        messages = state["messages"] if "messages" in state else []
        messages.append({"role": "system", "content": f"Decision made: {decision}"})
        output = f"Agent decision: {decision}"
        # Return the complete state dictionary
        return {"decision": decision, "input": input, "messages": messages, "output": output}

    def create_chain(self) -> Runnable:
        """
        Create and return a compiled state graph for this agent.

        Returns:
        --------
        CompiledStateGraph
            A compiled state graph representing the agent's workflow.
        """
        if self._chain is None:
            try:
                # Create the Workflow as StateGraph using the AgentState
                workflow = StateGraph(AgentState)
                # Add the nodes (start_agent, stock_agent, rag_agent)
                workflow.add_node("start", self._start_agent)
                decisions = {}
                for agent in self.agents:
                    workflow.add_node(
                        agent["agent"],
                        self._agent_node,
                        metadata={"agent_key": agent},
                    )
                    # Add the final edges to the END node
                    workflow.add_edge(agent["agent"], END)
                    decisions[agent["agent"]] = agent["condition"]

                # Add the conditional edge from start -> lamba (decision) -> defined agents
                workflow.add_conditional_edges(
                    "start",
                    lambda x: x["decision"],
                    decisions,
                )
                # Set the workflow entry point
                workflow.set_entry_point("start")
                # Compile the workflow
                self._chain = workflow.compile()

            except Exception as e:
                logger.error(f"Failed to create ConditionGraph: {e}")
                raise

        return self._chain
