"""
supervisor_agent.py

This module defines the SupervisorAgent class, an implementation of the supervisor pattern
for multi-agent coordination. The supervisor agent acts as a coordinator that can delegate
tasks to worker agents and manage the conversation flow between multiple specialized agents.

Classes:
--------
SupervisorState : TypedDict
    State definition for tracking delegation, worker responses, messages, and metadata.
SupervisorAgent : Agent
    Implements the supervisor pattern with task delegation to worker agents.

Dependencies:
-------------
- langchain_core: For message handling and prompt templates
- langgraph: For state graph workflow construction
"""

import os
from logging import getLogger
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from .agent import Agent

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = getLogger(__name__)
logger.setLevel(LOGLEVEL)


class SupervisorState(TypedDict):
    """
    State definition for the supervisor agent workflow.

    This state tracks messages, delegation decisions, worker responses, and metadata
    throughout the supervisor's coordination of multiple worker agents.

    Attributes:
    -----------
    messages : Annotated[Sequence[BaseMessage], add_messages]
        List of conversation messages with automatic message addition functionality.
    next_worker : str
        The worker agent selected to handle the current task.
    worker_response : Optional[str]
        The response from the delegated worker agent.
    delegation_reason : Optional[str]
        The reason why a specific worker was chosen for delegation.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_worker: str
    worker_response: Optional[str]
    delegation_reason: Optional[str]


class SupervisorAgent(Agent):
    """
    Implementation of a supervisor agent that coordinates multiple worker agents.

    The supervisor agent analyzes incoming requests, determines which worker agent
    is best suited to handle the task, delegates the work, and manages the response flow.
    This implements the supervisor pattern for multi-agent coordination.
    """

    def __init__(
        self,
        agent_key: str,
        workers: List[str],
        delegation_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Initialize a SupervisorAgent with worker coordination capabilities.

        Parameters:
        -----------
        agent_key : str
            Unique identifier for the agent.
        workers : List[str]
            List of worker agent keys that this supervisor can delegate to.
        delegation_prompt : str
            The prompt used to determine which worker should handle a task.
        model : str, optional
            The model to use for this agent. Defaults to "gpt-4o".
        temperature : float, optional
            The temperature setting for response generation. Defaults to 0.7.
        """
        super().__init__(agent_key, model, temperature)
        self.workers = workers
        self.delegation_prompt = delegation_prompt

    def get_system_prompts(self) -> List[Dict[str, str]]:
        """
        Get the system messages for this supervisor agent.

        Returns:
        --------
        List[Dict[str, str]]
            A list containing the supervisor's delegation prompt as a system message.
        """
        return [{"role": "system", "content": self.delegation_prompt}]

    def _should_continue(self, state: Dict[str, Any]) -> str:
        """
        Determine whether to continue with worker delegation or end the conversation.

        Parameters:
        -----------
        state : Dict[str, Any]
            Current state containing the supervisor's decision.

        Returns:
        --------
        str
            Either a worker agent key to delegate to, or 'END' to finish.
        """
        next_worker = state.get("next_worker", "END")
        if next_worker in self.workers:
            return next_worker
        return END

    def _supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supervisor decision node that determines which worker should handle the task.

        Parameters:
        -----------
        state : Dict[str, Any]
            Current state containing conversation messages.

        Returns:
        --------
        Dict[str, Any]
            Updated state with the delegation decision and reasoning.
        """
        # Create delegation prompt that includes available workers
        workers_list = ", ".join(self.workers)
        delegation_system_prompt = f"""You are a supervisor agent responsible for delegating tasks to worker agents.

Available workers: {workers_list}

Based on the conversation history, determine which worker agent should handle the current request.
If no delegation is needed (e.g., you can answer directly), respond with 'END'.

Respond in the following format:
WORKER: [worker_name or END]
REASON: [brief explanation of why this worker was chosen]

{self.delegation_prompt}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", delegation_system_prompt),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | self._get_azure_chat_openai()
        response = chain.invoke({"messages": state["messages"]})
        
        # Parse the response to extract worker and reason
        response_content = response.content if hasattr(response, 'content') else str(response)
        lines = response_content.strip().split('\n')
        
        next_worker = "END"
        delegation_reason = "No specific delegation needed"
        
        for line in lines:
            if line.startswith("WORKER:"):
                next_worker = line.replace("WORKER:", "").strip()
            elif line.startswith("REASON:"):
                delegation_reason = line.replace("REASON:", "").strip()
        
        logger.info(f"Supervisor delegating to: {next_worker}, reason: {delegation_reason}")
        
        return {
            "next_worker": next_worker,
            "delegation_reason": delegation_reason,
            "messages": state["messages"]
        }

    def _worker_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a worker agent based on the supervisor's delegation decision.

        Parameters:
        -----------
        state : Dict[str, Any]
            Current state containing the worker selection and conversation context.

        Returns:
        --------
        Dict[str, Any]
            Updated state with the worker's response.
        """
        from .agent_registry import agent_registry

        worker_key = state["next_worker"]
        worker_agent = agent_registry.get_agent(worker_key)
        
        if worker_agent is None:
            logger.error(f"Worker agent {worker_key} not found in registry.")
            error_message = f"Error: Worker agent '{worker_key}' not available."
            return {
                "messages": state["messages"] + [AIMessage(content=error_message)],
                "worker_response": error_message
            }

        # Get worker's system prompts and combine with conversation
        worker_messages = worker_agent.get_system_prompts() + state["messages"]
        worker_chain = worker_agent.create_chain()
        
        # Invoke the worker agent
        response = worker_chain.invoke({"messages": worker_messages})
        
        # Extract the response content
        if hasattr(response, 'content'):
            worker_response = response.content
        else:
            worker_response = str(response)
        
        # Add worker response to messages
        updated_messages = state["messages"] + [AIMessage(content=worker_response)]
        
        return {
            "messages": updated_messages,
            "worker_response": worker_response,
            "next_worker": "END"  # Reset for next iteration
        }

    def create_chain(self) -> Runnable:
        """
        Create and return a compiled state graph for the supervisor agent.

        Returns:
        --------
        CompiledStateGraph
            A compiled state graph representing the supervisor's workflow.
        """
        if self._chain is None:
            try:
                # Create the StateGraph
                workflow = StateGraph(SupervisorState)
                
                # Add the supervisor node
                workflow.add_node("supervisor", self._supervisor_node)
                
                # Add worker nodes
                for worker in self.workers:
                    workflow.add_node(worker, self._worker_node)
                    # Worker nodes go back to supervisor for potential re-delegation
                    workflow.add_edge(worker, "supervisor")
                
                # Add conditional edges from supervisor
                workflow.add_conditional_edges(
                    "supervisor",
                    self._should_continue,
                    {worker: worker for worker in self.workers} | {"END": END}
                )
                
                # Set entry point
                workflow.set_entry_point("supervisor")
                
                # Compile the workflow
                self._chain = workflow.compile()
                
            except Exception as e:
                logger.error(f"Failed to create SupervisorAgent workflow: {e}")
                raise

        return self._chain