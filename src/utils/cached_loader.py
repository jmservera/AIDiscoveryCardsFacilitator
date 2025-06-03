"""
Cached prompt file loader utility.

This module provides caching functionality for loading prompt files and creating
initial message structures for AI agents. It uses Streamlit's caching mechanism
to optimize file loading performance.

Functions:
----------
load_prompt_files : Cached function to load persona and document files
"""

from logging import getLogger
from typing import Dict, List, Optional, Union

logger = getLogger(__name__)


def load_prompt_files(
    persona_file_path: str, content_file_paths: Optional[Union[str, List[str]]] = None
) -> List[Dict[str, str]]:
    """
    Load content from prompt files and create initial messages.

    This function reads a persona file and optional content files to create
    a list of system messages. It automatically adds guardrails from the
    guardrails.md file and handles multiple content files.

    Parameters:
    -----------
    persona_file_path : str
        Path to the persona/system prompt file.
    content_file_paths : Optional[Union[str, List[str]]], optional
        Path to a single content/context file, or a list of file paths.
        If None, only the persona prompt will be loaded.

    Returns:
    --------
    List[Dict[str, str]]
        List of message objects with the system prompts loaded. Each message
        has 'role' and 'content' keys.

    Raises:
    -------
    FileNotFoundError
        If the persona file or any content file cannot be found.
    PermissionError
        If file access is denied.
    UnicodeDecodeError
        If file encoding is incompatible.
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
