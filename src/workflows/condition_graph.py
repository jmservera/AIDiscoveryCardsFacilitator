import os
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from streamlit.logger import get_logger
from typing_extensions import Annotated, TypedDict

from agents.agent_registry import agent_registry
from workflows.chat_graph import ChatGraph


class AgentState(TypedDict):
    input: str
    output: str
    decision: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


class ConditionGraph(ChatGraph):
    """
    LangGraph-based condition graph for multi-agent workflows.

    This class provides a LangGraph workflow for handling conditions and
    multi-agent interactions.
    """

    def __init__(
        self,
        condition: str,
        agents: List[str],
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
    ) -> None:
        """
        Initialize the ConditionGraph with configuration.

        Parameters:
        -----------
        condition : str
            The condition to evaluate for agent selection.
        agents : List[str]
            List of agent keys to include in the workflow.
        model : str, optional
            The model to use for chat completion. Defaults to "gpt-4o".
        temperature : float, optional
            The temperature setting for response generation. Defaults to 1.0.
        """
        self.condition = condition
        self.agents = agents
        super().__init__(model=model, temperature=temperature)

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
        response = chain.invoke({"input": state["input"]})
        # take the decision from the response
        decision = response.content.strip().lower()
        # Return the response for the next agent (decision and input required coming fron the Agent State)
        return {"decision": decision, "input": state["input"]}

    def _agent_node(self, state, config: RunnableConfig) -> Dict[str, Any]:
        """
        Create a node for the specified agent in the workflow.

        Parameters:
        -----------
        agent_key : str
            The key of the agent to create a node for.

        Returns:
        --------
        Dict[str, Any]
            A dictionary representing the agent node.
        """

        if "metadata" not in config or "agent_key" not in config["metadata"]:
            raise ValueError("Agent node requires 'metadata' with 'agent_key'")
        agent = agent_registry.get_agent(config["metadata"]["agent_key"])
        if not agent:
            raise ValueError(
                f"Agent with key '{config['metadata']['agent_key']}' not found in registry"
            )

        chain = (
            {
                "context": retriever_multimodal
                | RunnableLambda(ingestion.get_image_description),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(ingestion.multimodal_prompt)
            | rag_agent_llm
            | StrOutputParser()
        )

        return {"content": f"Agent node for {config['metadata']['agent_key']}"}

    def _create_graph(self) -> CompiledStateGraph:
        """
        Create the LangGraph workflow for condition-based agent selection.

        Returns:
        --------
        CompiledStateGraph
            Compiled LangGraph workflow for condition evaluation and agent interaction.
        """
        if self._graph is None:
            try:
                # Create the Workflow as StateGraph using the AgentState
                workflow = StateGraph(AgentState)
                # Add the nodes (start_agent, stock_agent, rag_agent)
                workflow.add_node("start", self._start_agent)
                workflow.add_node(
                    "facilitator_agent",
                    self._agent_node,
                    metadata={"agent_key": "facilitator"},
                )
                workflow.add_node(
                    "design_thinking_expert_agent",
                    self._agent_node,
                    metadata={"agent_key": "design_thinking_expert"},
                )
                workflow.add_node(
                    "ai_discovery_expert_agent",
                    self._agent_node,
                    metadata={"agent_key": "ai_discovery_expert"},
                )
                # Add the conditional edge from start -> lamba (decision) -> stock_agent or rag_agent
                workflow.add_conditional_edges(
                    "start",
                    lambda x: x["decision"],
                    {
                        "facilitator": "facilitator_agent",
                        "design_thinking_expert": "design_thinking_expert_agent",
                        "ai_discovery_expert": "ai_discovery_expert_agent",
                    },
                )
                # Set the workflow entry point
                workflow.set_entry_point("start")
                # Add the final edges to the END node
                workflow.add_edge("facilitator_agent", END)
                workflow.add_edge("design_thinking_expert_agent", END)
                workflow.add_edge("ai_discovery_expert_agent", END)
                # Compile the workflow
                self._graph = workflow.compile()
            except Exception as e:
                logger.error(f"Failed to create ConditionGraph: {e}")
                raise

        return self._graph

    async def create_chat_completion_async(
        self, messages: List[Dict[str, str]]
    ) -> AsyncIterator[Any]:
        """
        Create async chat completion with streaming support.

        This method provides a direct async interface for chat completion,
        replacing the synchronous wrapper approach.

        Parameters:
        -----------
        messages : List[Dict[str, str]]
            List of message dictionaries with 'role' and 'content' keys.

        Yields:
        -------
        Any
            Streaming response chunks compatible with LangChain format.
        """
        try:
            # Convert message dictionaries to LangChain format
            langchain_messages = self._convert_to_langchain_messages(messages)

            # Get the LLM for streaming

            value = await self._create_graph().ainvoke(
                {
                    "messages": langchain_messages,
                    "input": messages[-1].get("content", ""),
                },
            )
            yield value

        except Exception as e:
            logger.warning(f"LangGraph execution failed, using fallback response: {e}")
            # Fallback to mock response when Azure authentication fails
            async for chunk in self._create_async_fallback_response(messages):
                yield chunk
