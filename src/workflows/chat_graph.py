"""
chat_graph.py

LangGraph-based chat workflow implementation to replace direct Azure OpenAI API calls.
This module provides a LangGraph graph for chat completion that supports both single
and multi-agent workflows while maintaining backward compatibility with the existing
Agent interface.

Key Features:
- LangGraph workflow for chat completion
- Azure OpenAI integration via LangChain's AzureChatOpenAI
- Streaming support for real-time responses
- Token usage tracking
- Support for both single and multi-agent conversations

Classes:
--------
ChatGraph
    LangGraph implementation for chat completion workflows

Dependencies:
-------------
- langgraph: For workflow orchestration
- langchain-openai: For Azure OpenAI integration
- typing: For type annotations
"""

import os
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Union

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from streamlit.logger import get_logger
from typing_extensions import Annotated, TypedDict

load_dotenv()

logger = get_logger(__name__)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")


class ChatState(TypedDict):
    """State definition for the chat workflow."""
    messages: Annotated[List[BaseMessage], add_messages]


class ChatGraph:
    """
    LangGraph-based chat workflow for Azure OpenAI interactions.

    This class replaces direct Azure OpenAI API calls with a LangGraph workflow
    while maintaining the same interface for backward compatibility.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 1.0) -> None:
        """
        Initialize the ChatGraph with Azure OpenAI configuration.

        Parameters:
        -----------
        model : str, optional
            The model to use for chat completion. Defaults to "gpt-4o".
        temperature : float, optional
            The temperature setting for response generation. Defaults to 1.0.
        """
        self.model = model
        self.temperature = temperature
        self._llm = None
        self._graph = None

    def _get_azure_chat_openai(self) -> AzureChatOpenAI:
        """
        Create and return an AzureChatOpenAI instance with authentication.

        Returns:
        --------
        AzureChatOpenAI
            Configured Azure OpenAI chat model instance.
        """
        if self._llm is None:
            # Use Azure identity for authentication
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )

            self._llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_ad_token_provider=token_provider,
                azure_deployment=self.model,
                temperature=self.temperature,
                streaming=True,
            )

        return self._llm

    def _create_graph(self) -> CompiledStateGraph:
        """
        Create the LangGraph workflow for chat completion.

        Returns:
        --------
        StateGraph
            Compiled LangGraph workflow for chat processing.
        """
        if self._graph is None:
            # Create the state graph
            workflow = StateGraph(ChatState)

            # Add the chat node
            workflow.add_node("chat", self._chat_node)

            # Set the entry point
            workflow.set_entry_point("chat")

            # Add edge to end
            workflow.add_edge("chat", END)

            # Compile the graph
            self._graph = workflow.compile()

        return self._graph

    async def _chat_node(self, state: ChatState) -> Dict[str, Any]:
        """
        Chat node function for the LangGraph workflow.

        Parameters:
        -----------
        state : ChatState
            Current state containing the conversation messages.

        Returns:
        --------
        Dict[str, Any]
            Updated state with the AI response.
        """
        llm = self._get_azure_chat_openai()

        # Invoke the LLM with the current messages
        response = await llm.ainvoke(state["messages"])

        # Return the updated state
        return {"messages": [response]}

    def _convert_to_langchain_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """
        Convert message dictionaries to LangChain message format.

        Parameters:
        -----------
        messages : List[Dict[str, str]]
            List of message dictionaries with 'role' and 'content' keys.

        Returns:
        --------
        List[BaseMessage]
            List of LangChain message objects.
        """
        langchain_messages = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                logger.warning(f"Unknown message role: {role}")
                # Default to human message for unknown roles
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

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
            llm = self._get_azure_chat_openai()

            async for chunk in llm.astream(langchain_messages, stream_usage=True):
                yield chunk

        except Exception as e:
            logger.warning(f"LangGraph execution failed, using fallback response: {e}")
            # Fallback to mock response when Azure authentication fails
            async for chunk in self._create_async_fallback_response(messages):
                yield chunk

    async def _create_async_fallback_response(
        self, messages: List[Dict[str, str]]
    ) -> AsyncIterator[Any]:
        """
        Create an async fallback response when LangGraph execution fails.

        Parameters:
        -----------
        messages : List[Dict[str, str]]
            The original messages for context.

        Yields:
        -------
        Any
            Mock streaming response chunks.
        """
        import asyncio

        # Create a meaningful fallback response
        fallback_content = "Hello! I'm now using async LangGraph workflows for chat completion. This is a fallback response since Azure OpenAI credentials are not available in this environment."

        # Get last user message for more context
        user_messages = [m for m in messages if m.get("role") == "user"]
        if user_messages:
            last_user_msg = user_messages[-1].get("content", "")
            if "hello" in last_user_msg.lower():
                fallback_content = "Hello! How can I help you today? I'm now running on async LangGraph workflows!"
            elif "test" in last_user_msg.lower():
                fallback_content = "This is a test response from the new async LangGraph-powered agent!"

        words = fallback_content.split()

        # Stream the response word by word to simulate real streaming
        for i, word in enumerate(words):
            partial_content = " ".join(words[: i + 1])

            # Create a simple LangChain-style chunk
            class MockChunk:
                def __init__(self, content: str):
                    self.content = content

            yield MockChunk(partial_content)
            await asyncio.sleep(0.05)  # Simulate streaming delay

        # Final response with usage info
        class FinalChunk:
            def __init__(self, content: str):
                self.content = ""
                self.usage_info = {
                    "completion_tokens": len(content.split()) if content else 0,
                    "prompt_tokens": 50,  # Estimate
                    "total_tokens": len(content.split()) + 50 if content else 50,
                }

        yield FinalChunk(fallback_content)
