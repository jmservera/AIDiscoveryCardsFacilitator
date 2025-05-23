"""
Utility functions for interacting with Azure OpenAI API

This module provides a set of utilities to authenticate, communicate with, and process
responses from Azure OpenAI services. It handles token management, message formatting,
and streaming chat completions using Azure OpenAI's API.

Key Features:
- Authentication with Azure OpenAI using DefaultAzureCredential
- Token counting and management for input and output messages
- Streaming chat completions with proper error handling
- Loading and formatting system prompts from files
- XML tag detection and handling for document embedding

Environment Variables:
- AZURE_OPENAI_ENDPOINT: The endpoint URL for Azure OpenAI
- AZURE_OPENAI_DEPLOYMENT_NAME: The deployment name to use (defaults to "gpt-4o")

Dependencies:
- openai: Main client for interacting with OpenAI API
- azure.identity: For authentication with Azure
- streamlit: For UI components and session management
- tiktoken: For token counting
"""

import os
import re
from typing import Dict, List, Optional, Union

import openai
import streamlit as st
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from st_copy import copy_button
from streamlit.logger import get_logger

# Configure logging using Streamlit's logger
logger = get_logger(__name__)

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")


@st.cache_resource
def get_client() -> openai.AzureOpenAI:
    """Get the Azure OpenAI client using DefaultAzureCredential.

    Returns:
        An authenticated Azure OpenAI client
    """
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    return openai.AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2024-06-01",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def create_chat_completion(messages: List[Dict[str, str]]) -> openai.Stream:
    """Create and return a new chat completion request.

    Args:
        messages: List of message objects with role and content

    Returns:
        A streaming response from Azure OpenAI
    """

    client = get_client()

    # Create and return a new chat completion request
    return client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        stream=True,
        stream_options={"include_usage": True},
    )


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


def handle_chat_prompt(prompt: str, page: Dict) -> None:
    """Process a user prompt, send to Azure OpenAI and display the response.

    Args:
        prompt: The user's text input
        page: Dictionary containing page state including messages history

    Returns:
        None - updates the session state and UI directly
    """
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

    # Send the user's prompt to Azure OpenAI and display the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        completion = None
        for response in create_chat_completion(page["messages"]):
            if response.choices:
                try:
                    if response.choices[0].delta is not None:
                        full_response += response.choices[0].delta.content or ""
                        message_placeholder.markdown(full_response + "▌")
                    else:
                        logger.debug(response.choices[0].model_dump_json())
                except (AttributeError, IndexError) as e:
                    logger.exception("Error processing response: %s", e)
                    full_response += "An error happened, retry your request.\n"
            completion = response
        message_placeholder.markdown(full_response)

    # Add the response to the messages
    page["messages"].append({"role": "assistant", "content": full_response})

    # Display token usage
    if completion and completion.usage:
        copy_button(full_response, key=full_response)
        st.caption(
            f"""Token usage for this interaction:
        - Input tokens: {input_tokens}
        - Output tokens: {completion.usage.completion_tokens}
        - Total tokens: {completion.usage.total_tokens}"""
        )


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
    # Read the persona file
    with open(persona_file_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
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
