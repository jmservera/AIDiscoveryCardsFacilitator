from typing import Dict, List, Optional, Union

import streamlit as st
from streamlit.logger import get_logger

logger = get_logger(__name__)


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
