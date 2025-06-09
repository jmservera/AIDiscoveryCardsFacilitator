"""
react_agent.py

This module defines the ReactAgent class, an implementation of the ReAct (Reasoning and Acting)
pattern for conversational AI agents. The ReAct agent combines reasoning with action-taking
capabilities, allowing it to think through problems step by step and take appropriate actions.

Classes:
--------
ReactState : TypedDict
    State definition for tracking thoughts, actions, observations, and messages.
ReactAgent : Agent
    Implements the ReAct pattern with reasoning and action capabilities.

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


class ReactState(TypedDict):
    """
    State definition for the ReAct agent workflow.

    This state tracks the reasoning process, actions taken, observations made,
    and conversation messages throughout the ReAct agent's execution.

    Attributes:
    -----------
    messages : Annotated[Sequence[BaseMessage], add_messages]
        List of conversation messages with automatic message addition functionality.
    thought : Optional[str]
        The current reasoning/thought process of the agent.
    action : Optional[str]
        The action the agent has decided to take.
    observation : Optional[str]
        The observation or result from the action taken.
    iteration_count : int
        Number of reasoning iterations performed.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    thought: Optional[str]
    action: Optional[str]
    observation: Optional[str]
    iteration_count: int


class ReactAgent(Agent):
    """
    Implementation of a ReAct (Reasoning and Acting) agent.

    The ReAct agent follows a structured approach to problem-solving by alternating
    between reasoning (thinking about the problem) and acting (taking specific actions
    or providing responses). This pattern helps create more thoughtful and systematic
    responses to complex queries.
    """

    def __init__(
        self,
        agent_key: str,
        persona: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: int = 3,
    ) -> None:
        """
        Initialize a ReactAgent with reasoning and action capabilities.

        Parameters:
        -----------
        agent_key : str
            Unique identifier for the agent.
        persona : str
            Path to the persona prompt file that defines the agent's behavior.
        model : str, optional
            The model to use for this agent. Defaults to "gpt-4o".
        temperature : float, optional
            The temperature setting for response generation. Defaults to 0.7.
        max_iterations : int, optional
            Maximum number of reasoning iterations. Defaults to 3.
        """
        super().__init__(agent_key, model, temperature)
        self.persona = persona
        self.max_iterations = max_iterations

    def get_system_prompts(self) -> List[Dict[str, str]]:
        """
        Get the system messages for this ReAct agent.

        Returns:
        --------
        List[Dict[str, str]]
            A list of system messages including the persona and ReAct instructions.
        """
        from utils.cached_loader import load_prompt_files
        
        base_prompts = load_prompt_files(self.persona, None)
        
        # Add ReAct pattern instructions
        react_instructions = {
            "role": "system",
            "content": """You are a ReAct (Reasoning and Acting) agent. Follow this pattern:

1. THOUGHT: Think about the user's request and what you need to do
2. ACTION: Decide on the specific action or response you will provide
3. OBSERVATION: Reflect on your action and its effectiveness

Format your response as:
THOUGHT: [your reasoning about the request]
ACTION: [your specific response or action]
OBSERVATION: [reflection on your response]

Be thorough in your reasoning and provide helpful, well-considered responses."""
        }
        
        return base_prompts + [react_instructions]

    def _should_continue(self, state: Dict[str, Any]) -> str:
        """
        Determine whether to continue reasoning or provide final response.

        Parameters:
        -----------
        state : Dict[str, Any]
            Current state containing iteration count and reasoning progress.

        Returns:
        --------
        str
            Either 'think' to continue reasoning or 'respond' to provide final answer.
        """
        iteration_count = state.get("iteration_count", 0)
        
        # If we've reached max iterations or have a clear action, move to response
        if iteration_count >= self.max_iterations or state.get("action"):
            return "respond"
        return "think"

    def _think_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reasoning node where the agent thinks about the problem.

        Parameters:
        -----------
        state : Dict[str, Any]
            Current state containing conversation messages.

        Returns:
        --------
        Dict[str, Any]
            Updated state with reasoning thoughts.
        """
        iteration_count = state.get("iteration_count", 0)
        
        think_prompt = ChatPromptTemplate.from_messages([
            ("placeholder", "{messages}"),
            ("system", f"""This is reasoning iteration {iteration_count + 1}/{self.max_iterations}.
            
Think step by step about the user's request. Consider:
- What is the user really asking for?
- What information do you need to provide a helpful response?
- What approach would be most effective?

Provide your thoughts in this format:
THOUGHT: [your detailed reasoning]""")
        ])
        
        chain = think_prompt | self._get_azure_chat_openai()
        response = chain.invoke({"messages": state["messages"]})
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract thought from response
        thought = ""
        for line in response_content.split('\n'):
            if line.startswith("THOUGHT:"):
                thought = line.replace("THOUGHT:", "").strip()
                break
        
        return {
            "messages": state["messages"],
            "thought": thought,
            "iteration_count": iteration_count + 1,
            "action": state.get("action"),
            "observation": state.get("observation")
        }

    def _respond_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Response node where the agent provides the final answer.

        Parameters:
        -----------
        state : Dict[str, Any]
            Current state containing reasoning and conversation context.

        Returns:
        --------
        Dict[str, Any]
            Updated state with the final response.
        """
        # Include the thought process in the prompt for better responses
        thought_context = f"Previous reasoning: {state.get('thought', 'No previous reasoning')}"
        
        respond_prompt = ChatPromptTemplate.from_messages([
            ("placeholder", "{messages}"),
            ("system", f"""{thought_context}

Now provide your final response using the ReAct format:
THOUGHT: [final reasoning about your response]
ACTION: [your actual response to the user]
OBSERVATION: [reflection on how well this addresses the user's needs]

Make sure your ACTION contains a complete and helpful response to the user's question.""")
        ])
        
        chain = respond_prompt | self._get_azure_chat_openai()
        response = chain.invoke({"messages": state["messages"]})
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract components from response
        final_thought = ""
        action = ""
        observation = ""
        
        lines = response_content.split('\n')
        for line in lines:
            if line.startswith("THOUGHT:"):
                final_thought = line.replace("THOUGHT:", "").strip()
            elif line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip()
            elif line.startswith("OBSERVATION:"):
                observation = line.replace("OBSERVATION:", "").strip()
        
        # Create the final response message combining all components
        final_response = f"**Reasoning:** {final_thought}\n\n**Response:** {action}\n\n**Reflection:** {observation}"
        
        updated_messages = state["messages"] + [AIMessage(content=final_response)]
        
        return {
            "messages": updated_messages,
            "thought": final_thought,
            "action": action,
            "observation": observation,
            "iteration_count": state.get("iteration_count", 0)
        }

    def create_chain(self) -> Runnable:
        """
        Create and return a compiled state graph for the ReAct agent.

        Returns:
        --------
        CompiledStateGraph
            A compiled state graph representing the ReAct agent's workflow.
        """
        if self._chain is None:
            try:
                # Create the StateGraph
                workflow = StateGraph(ReactState)
                
                # Add nodes
                workflow.add_node("think", self._think_node)
                workflow.add_node("respond", self._respond_node)
                
                # Add edges
                workflow.add_edge("respond", END)
                
                # Add conditional edge from think
                workflow.add_conditional_edges(
                    "think",
                    self._should_continue,
                    {
                        "think": "think",
                        "respond": "respond"
                    }
                )
                
                # Set entry point
                workflow.set_entry_point("think")
                
                # Compile the workflow
                self._chain = workflow.compile()
                
            except Exception as e:
                logger.error(f"Failed to create ReactAgent workflow: {e}")
                raise

        return self._chain