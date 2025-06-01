"""
Utility functions for interacting with Azure OpenAI API

This module provides a set of utilities to authenticate, communicate with, and process
responses from Azure OpenAI services. It handles token management, message formatting,
and streaming chat completions.

This module has been updated to work with LangGraph-based chat workflows.
The Agent classes use LangGraph and LangChain's AzureChatOpenAI instead of direct
openai.AzureOpenAI client usage.

Key Features:
- Token counting and management for input and output messages
- Streaming chat completions with proper error handling via LangGraph workflows
- Loading and formatting system prompts from files
- XML tag detection and handling for document embedding

Environment Variables:
- AZURE_OPENAI_ENDPOINT: The endpoint URL for Azure OpenAI


Dependencies:
- agents.agent: Uses LangGraph-powered Agent class
- streamlit: For UI components and session management
- tiktoken: For token counting
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import streamlit as st
import tiktoken
from st_copy import copy_button
from streamlit.logger import get_logger
from streamlit_mermaid import st_mermaid

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from agents.agent import Agent

# Configure logging using Streamlit's logger
logger = get_logger(__name__)


def count_tokens(messages: List[Dict[str, str]]) -> int:
    """Count the number of tokens in the messages.

    Args:
        messages: List of message objects with role and content

    Returns:
        Number of tokens in the messages
    """
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )  # This is the encoding used by GPT-4
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for _, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens


def count_xml_tags(text: str) -> int:
    """Count the number of XML tags in a string.

    Args:
        text: String to count XML tags in

    Returns:
        Number of XML tags in the string
    """
    # Define the regex pattern for XML tags
    pattern = r"<[^>]+>"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Return the number of matches
    return len(matches)


def handle_chat_prompt(
    prompt: str, messages: List[Dict[str, str]], agent: "Agent"
) -> None:
    """Process a user prompt, send to Azure OpenAI via async LangGraph and display the response.

    Args:
        prompt: The user's text input
        messages: List of chat messages
        agent: Agent instance to handle the conversation

    Returns:
        None - updates the session state and UI directly
    """
    import asyncio

    async def async_handle_chat_prompt():
        """Async implementation of chat prompt handling."""
        # Calculate tokens in the input
        input_tokens = count_tokens(messages)
        logger.debug("Input tokens: %d", input_tokens)

        # Send the user's prompt to Azure OpenAI via async LangGraph and display the response
        message_placeholder = st.empty()
        message_placeholder.markdown("*Generating response...*")
        full_response = ""
        final_chunk = None
        try:
            logger.debug(
                "Creating async chat completion using agent %s with model %s and temperature %s",
                agent.agent_key,
                agent.model,
                agent.temperature,
            )
            async for chunk in agent.create_chat_completion_async(messages):
                if isinstance(chunk, tuple):
                    # If the chunk is a tuple, it means we received multiple messages
                    # This is typical for LangChain streaming responses
                    logger.debug("Received chunk as tuple of messages: %s", chunk)
                    for msg in chunk:
                        if hasattr(msg, "content") and msg.content:
                            full_response += msg.content
                            message_placeholder.markdown(full_response + "▌")
                        elif hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            final_chunk = msg
                    else:
                        logger.warning("Received unexpected chunk format: %s", chunk)
                else:
                    # Process LangChain streaming chunks
                    if hasattr(chunk, "content") and chunk.content:
                        # Regular streaming chunk with content
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                    elif hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        # Final chunk with usage information
                        final_chunk = chunk
                    else:
                        logger.warning("Received unexpected chunk format: %s", chunk)
        except Exception as e:
            logger.exception("Error during async chat completion: %s", e)
            full_response += "An error happened, retry your request.\n"
        finally:
            # Ensure we always clear the placeholder even if an error occurs
            if not full_response:
                st.markdown("Error: No response received from the model, try again.")

        logger.debug("Full response before rendering: %s", full_response)
        # Clear the placeholder after streaming
        message_placeholder.empty()

        # Render the response text with potential Mermaid diagrams
        render_message(full_response)

        return full_response, final_chunk

    # Cleanup prompt
    if count_xml_tags(prompt) > 0:
        logger.debug("Prompt contains XML tags.")
        # embed documents to avoid harm
        prompt = f"<documents>{prompt}</documents>"

    # Echo the user's prompt to the chat window
    messages.append({"role": "user", "content": prompt})
    logger.debug("Writing user prompt to chat")
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute the async chat handling within the assistant context
    with st.chat_message("assistant"):
        try:
            # Try to run the async function directly
            # In most Streamlit contexts, this should work fine
            full_response, final_chunk = asyncio.run(async_handle_chat_prompt())
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                logger.warning(
                    "Already in event loop, using fallback synchronous handling"
                )
                # If we're already in an event loop, fall back to a simpler approach
                # This should be rare in normal Streamlit usage
                message_placeholder = st.empty()
                message_placeholder.markdown("*Generating response...*")

                # Simple fallback response
                full_response = (
                    "The async agent is temporarily unavailable. Please try again."
                )
                final_chunk = None

                message_placeholder.empty()
                render_message(full_response)
            else:
                # Re-raise other RuntimeErrors
                raise

    # Add the response to the messages
    messages.append({"role": "assistant", "content": full_response})
    copy_button(full_response, key=hex(hash(full_response)))

    # Display token usage
    if final_chunk and hasattr(final_chunk, "usage_metadata"):
        usage = final_chunk.usage_metadata
        logger.info(usage)
        st.caption(
            f"""Token usage for this interaction:
        - Input tokens: {usage['input_tokens']}
        - Output tokens: {usage['output_tokens']}
        - Total tokens: {usage['total_tokens']}"""
        )


@st.cache_data
def load_prompt_files(
    persona_file_path: str, content_file_paths: Optional[Union[str, List[str]]] = None
) -> List[Dict[str, str]]:
    """Load content from prompt files and create initial messages.

    Args:
        persona_file_path: Path to the persona/system prompt file
        content_file_paths: Path to a single content/context file, or a list of file paths.
                          If None, only the persona prompt will be loaded.

    Returns:
        List of message objects with the system prompts loaded
    """
    logger.debug("Loading system messages from persona file: %s", persona_file_path)
    # Read the persona file
    with open(persona_file_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    logger.debug("Adding guardrails to system prompt")
    # Add security instructions to the system prompt
    with open("prompts/guardrails.md", "r", encoding="utf-8") as f:
        system_prompt += f.read()

    messages = [{"role": "system", "content": system_prompt}]

    # Handle content file(s)
    if content_file_paths:
        # Convert single string to list for uniform handling
        if isinstance(content_file_paths, str):
            content_file_paths = [content_file_paths]

        # Process each content file
        for file_path in content_file_paths:
            try:
                logger.debug("Loading content from file: %s", file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    system_document = f.read()
                    messages.append(
                        {
                            "role": "system",
                            "content": f"\n<documents>{system_document}</documents>",
                        }
                    )
            except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
                logger.exception("Error loading document file %s: %s", file_path, e)

    return messages


def extract_mermaid_diagrams(text: str) -> List[Tuple[str, str]]:
    """Extract Mermaid diagrams from markdown text.

    Args:
        text: Markdown text that may contain Mermaid diagrams

    Returns:
        List of tuples containing (full_match, diagram_code) where full_match
        is the complete match including the markdown code block markers,
        and diagram_code is just the Mermaid diagram content
    """
    # Pattern to match ```mermaid ... ``` blocks (case insensitive for "mermaid")
    pattern = r"```(?i:mermaid)\s+([\s\S]+?)```"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Return list of (full_match, diagram_code) tuples
    results = []
    for match in matches:
        # Use a case insensitive search to find the exact full match in the original text
        full_match_pattern = rf"```(?i:mermaid)\s+{re.escape(match)}\s*```"
        full_match_search = re.search(full_match_pattern, text)
        if full_match_search:
            full_match = full_match_search.group(0)
        else:
            full_match = f"```mermaid\n{match}\n```"

        results.append((full_match, match.strip()))

    return results


def render_message(message: str) -> None:
    """Render the Markdown text.
       Supports Mermaid diagrams.

    Args:
        message: The full text response that may contain Mermaid diagrams

    Returns:
        None - renders the content directly to the Streamlit UI
    """
    logger.debug("Rendering response text with potential Mermaid diagrams")
    # Extract mermaid diagrams from the response
    mermaid_diagrams = extract_mermaid_diagrams(message)

    if not mermaid_diagrams:
        # If no mermaid diagrams, just display the full response
        st.markdown(message)
        return

    logger.info("Found %d mermaid diagrams in the response", len(mermaid_diagrams))
    # Process text with mermaid diagrams
    remaining_text = message

    for full_match, diagram_code in mermaid_diagrams:
        # Split the text at the diagram position
        parts = remaining_text.split(full_match, 1)

        # Display the text before the diagram
        if parts[0]:
            st.markdown(parts[0])

        # Display the mermaid diagram
        try:
            # Determine diagram type to better calculate height
            diagram_type = "unknown"
            if diagram_code.strip().startswith(
                "graph"
            ) or diagram_code.strip().startswith("flowchart"):
                diagram_type = "flowchart"
            elif diagram_code.strip().startswith("sequenceDiagram"):
                diagram_type = "sequence"
            elif diagram_code.strip().startswith("classDiagram"):
                diagram_type = "class"
            elif diagram_code.strip().startswith("gantt"):
                diagram_type = "gantt"
            elif diagram_code.strip().startswith("pie"):
                diagram_type = "pie"
            elif diagram_code.strip().startswith("journey"):
                diagram_type = "journey"

            # Calculate complexity metrics
            line_count = diagram_code.count("\n") + 1
            node_count = (
                diagram_code.count("[")
                + diagram_code.count("{")
                + diagram_code.count("(")
            )
            connection_count = (
                diagram_code.count("-->")
                + diagram_code.count("==>")
                + diagram_code.count("-.->")
            )

            # Get user's diagram scale preference
            scale_factor = get_diagram_scale_factor()

            # Adjust height based on diagram type and complexity
            if diagram_type == "flowchart":
                # Flowcharts can expand horizontally or vertically, depending on the type
                # Possible FlowChart orientations are:
                #   TB - Top to bottom
                #   TD - Top-down/ same as top to bottom
                #   BT - Bottom to top
                #   RL - Right to left
                #   LR - Left to right

                # Check for orientation in the diagram
                if any(orientation in diagram_code for orientation in ["RL", "LR"]):
                    # Horizontal flowcharts need less height but more width consideration
                    height = max(300, 3000 // line_count)
                    # But, if they contain subgraphs or complex nodes,
                    # we still need a minimum height
                else:
                    # Vertical flowcharts (TB, TD, BT) need more height
                    height = max(node_count * 80, line_count * 25, 400)

            elif diagram_type == "sequence":
                # Sequence diagrams need more height per interaction
                height = max(line_count * 30, connection_count * 60, 500)
            elif diagram_type == "class":
                # Class diagrams tend to be taller
                height = max(node_count * 100, line_count * 25, 600)
            elif diagram_type == "gantt":
                # Gantt charts need height based on tasks
                task_count = diagram_code.count(":")
                height = max(task_count * 60, line_count * 20, 400)
            elif diagram_type == "pie":
                # Pie charts are generally more compact
                height = max(line_count * 20, 400)
            elif diagram_type == "journey":
                # Journey diagrams height is inversely proportional to length
                height = max(300, 3000 // line_count)
            else:
                # Default calculation for unknown types
                height = max(line_count * 30, node_count * 70, 500)

            # Apply user scaling preference
            height = int(height * scale_factor)

            # Apply user-defined scaling factor
            scale_factor = get_diagram_scale_factor()
            height = int(height * scale_factor)

            tab1, tab2 = st.tabs(["Diagram", "Code"])
            with tab1:
                st_mermaid(
                    diagram_code,
                    height=str(height),
                    width="container",
                    pan=False,
                    zoom=False,
                )
            with tab2:
                # Use container width to adapt to page size
                st.markdown(f"```mermaid\n{diagram_code}\n```")
                copy_button(diagram_code, key=diagram_code)
        except Exception as e:
            logger.exception("Error rendering mermaid diagram: %s", e)
            st.error(f"Failed to render diagram: {e}")
            st.code(diagram_code, language="mermaid")

        # Update remaining text
        if len(parts) > 1:
            remaining_text = parts[1]
        else:
            remaining_text = ""

    # Display any remaining text after the last diagram
    if remaining_text:
        st.markdown(remaining_text)


def get_diagram_scale_factor() -> float:
    """Get the user's preferred diagram scaling factor from session state.

    Returns:
        A float representing the scaling factor for diagram sizes (default 1.0)
    """
    # Initialize scale factor if not already in session state
    if "diagram_scale_factor" not in st.session_state:
        st.session_state.diagram_scale_factor = 1.0

    return st.session_state.diagram_scale_factor


def set_diagram_scale_factor(scale_factor: float) -> None:
    """Set the user's preferred diagram scaling factor in session state.

    Args:
        scale_factor: A float representing the scaling factor for diagrams
    """
    st.session_state.diagram_scale_factor = scale_factor
