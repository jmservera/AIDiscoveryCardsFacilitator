"""
Utility functions for interacting with language models (legacy module)

This module is being deprecated in favor of the agent_core module. It now provides
compatibility functions that delegate to agent_core for LLM interactions.

Key Features:
- Legacy token counting and management for input and output messages
- XML tag detection and handling for document embedding
- Compatibility layer for the agent_core module

Dependencies:
- agent_core: For delegating LLM interactions
- streamlit: For UI components and session management
"""

import logging
import re
from typing import Dict, List, Optional, Union

import streamlit as st
from streamlit.logger import get_logger

from agent_core import agent_core, load_prompt_files, count_tokens, count_xml_tags

# Configure logging using Streamlit's logger
logger = get_logger(__name__)


def create_chat_completion(messages: List[Dict[str, str]]):
    """Create and return a new chat completion request.
    
    This is a compatibility function that delegates to agent_core.
    
    Args:
        messages: List of message objects with role and content
        
    Returns:
        A streaming response from the LLM
    """
    logger.warning(
        "The openai_utils.create_chat_completion function is deprecated. "
        "Please use agent_core.create_chat_completion instead."
    )
    return agent_core.create_chat_completion(messages)


def handle_chat_prompt(prompt: str, page: Dict) -> None:
    """Process a user prompt, send to LLM and display the response.
    
    This is a compatibility function that delegates to agent_core.

    Args:
        prompt: The user's text input
        page: Dictionary containing page state including messages history

    Returns:
        None - updates the session state and UI directly
    """
    # This function needs to maintain UI compatibility with the old version
    # Cleanup prompt
    if count_xml_tags(prompt) > 0:
        logger.debug("Prompt contains XML tags.")
        # embed documents to avoid harm
        prompt = f"<documents>{prompt}</documents>"

    # Echo the user's prompt to the chat window
    page["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Calculate tokens in the input
    input_tokens = count_tokens(page["messages"])

    # Send the user's prompt to LLM and display the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        completion = None
        
        try:
            for response in agent_core.create_chat_completion(page["messages"]):
                if response.get("choices") and len(response["choices"]) > 0:
                    delta = response["choices"][0].get("delta", {})
                    if delta and delta.get("content") is not None:
                        full_response += delta["content"] or ""
                        message_placeholder.markdown(full_response + "▌")
                    else:
                        logger.debug(str(response))
                completion = response
            message_placeholder.markdown(full_response)
        except Exception as e:
            logger.exception("Error processing response: %s", e)
            full_response += "An error happened, retry your request.\n"
            message_placeholder.markdown(full_response)

    # Add the response to the messages
    page["messages"].append({"role": "assistant", "content": full_response})

    # Display token usage
    if completion and completion.get("usage"):
        from st_copy import copy_button
        copy_button(full_response, key=full_response)
        st.caption(
            f"""Token usage for this interaction:
        - Input tokens: {input_tokens}
        - Output tokens: {completion["usage"].completion_tokens}
        - Total tokens: {completion["usage"].total_tokens}"""
        )
