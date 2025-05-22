"""
Streamlit page components for OpenAI-powered chat interfaces

This module provides components and utilities for building interactive chat interfaces that
communicate with Azure OpenAI. It handles the UI/UX aspects of the chat experience and
delegates the AI interactions to the openai_utils module.

Key Features:
- Cached system message initialization
- Dynamic chat page generation with customizable personas
- Support for both single-agent and multi-agent pages
- Persistent chat history across page reloads
- Integration with OpenAI's chat completion API

Page Types:
- agent_page: Single agent chat interface with one persona and document context
- multiagent_page: Multi-agent chat interface with multiple personas

Dependencies:
- streamlit: For UI components and session management
- st_copy: For adding copy buttons to chat messages
- utils.openai_utils: For OpenAI interactions and prompt management

The module works by creating agent pages with specific personas and document contexts
that can be integrated into a multi-page Streamlit application.
"""

import streamlit as st
from st_copy import copy_button

from agent_registry import agent_registry
from utils.openai_utils import handle_chat_prompt, load_prompt_files


@st.cache_data
def get_system_messages(persona: str, documents=None):
    """
    Return the initial messages for the chat history.

    Parameters:
    -----------
    persona : str
        Path to the persona prompt file.
    documents : str or list[str], optional
        Path(s) to document file(s) to use as context. Can be a single string or a list of strings.

    Returns:
    --------
    list
        List of message objects with the system prompts loaded.
    """
    return load_prompt_files(persona, documents)


@st.cache_data
def get_system_messages_multiagent(personas: list[str], documents: list[str] = None):
    """
    Return the initial messages for a multi-agent chat history.

    Parameters:
    -----------
    personas : list[str]
        List of paths to persona prompt files.
    documents : list[str], optional
        List of paths to document files. If provided, each document is paired with the
        corresponding persona. If a persona needs multiple documents, provide a list of lists.

    Returns:
    --------
    list
        Combined list of message objects for all personas with their documents.
    """
    messages = []
    for i, persona in enumerate(personas):
        # Handle document pairing for this persona
        persona_docs = None
        if documents and i < len(documents):
            persona_docs = documents[i]

        persona_messages = load_prompt_files(persona, persona_docs)
        messages.extend(persona_messages)
    return messages


def multiagent_page(
    personas: list[str], title: str, subtitle: str, documents: list[str] = None
):
    """
    Create a Streamlit chat page for interacting with multiple agent personas.

    This function returns a page function that sets up a chatbot interface with multiple
    personas. The page includes a title, subtitle, and a chat history that persists across
    page reloads by storing it in the Streamlit session state.

    Parameters:
    -----------
    personas : list[str]
        A list of persona or role prompt file names that the assistant should adopt in the
        conversation.
    title : str
        The title to display at the top of the page.
    subtitle : str
        The subtitle to display below the title.
    documents : list[str], optional
        A list of document or context files that the assistant should use as reference.
        If provided, each document is paired with the corresponding persona.

    Returns:
    --------
    function
        A page function that can be called to render the chat interface.

    Notes:
    ------
    The function relies on several external dependencies:
    - get_system_messages_multiagent(): For initializing the chat with system messages
    - handle_chat_prompt(): From the openai_utils module.
                            Expected to process user inputs and generate responses
    - copy_button(): Expected to add a copy button to assistant messages

    The chat history is stored in st.session_state.pages keyed by the current URL.
    """

    def page():
        st.title(title)
        st.subheader(subtitle)

        if "pages" not in st.session_state:
            st.session_state.pages = {}

        if st.context.url not in st.session_state.pages:
            st.session_state.pages[st.context.url] = {}

        page = st.session_state.pages[st.context.url]
        # Initialize chat history
        if "messages" not in page:
            page["messages"] = get_system_messages_multiagent(personas, documents)

        # Display chat messages from history on app rerun
        for message in page["messages"]:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    msg = message["content"]
                    st.markdown(msg)
                    if message["role"] == "assistant":
                        copy_button(msg, key=msg)

        # Await a user message and handle the chat prompt when it comes in.
        if prompt := st.chat_input("Enter a message:"):
            handle_chat_prompt(prompt, page)

    return page


def agent_page(persona: str, documents=None, header: str = None, subtitle: str = None):
    """
    Create a Streamlit chat page for interacting with an agent persona.

    This function returns a page function that sets up a chatbot interface with a specific
    persona and document context. The page includes a header, subtitle, and a chat history
    that persists across page reloads by storing it in the Streamlit session state.

    Parameters:
    -----------
    persona : str
        The persona or role prompt file name that the assistant should adopt in the conversation.
    documents : str or list[str], optional
        The document or context file(s) that the assistant should use as reference.
        Can be a single file path or a list of file paths.
    header : str
        The header text to display at the top of the page.
    subtitle : str
        The subtitle text to display below the header.

    Returns:
    --------
    function
        A page function that can be called to render the chat interface.

    Notes:
    ------
    The function relies on several external dependencies:
    - get_system_messages(): For initializing the chat with system messages
    - handle_chat_prompt(): From the openai_utils module.
                            Expected to process user inputs and generate responses
    - copy_button(): Expected to add a copy button to assistant messages

    The chat history is stored in st.session_state.pages keyed by the current URL.
    """

    def page():
        st.title(header)
        st.subheader(subtitle)

        if "pages" not in st.session_state:
            st.session_state.pages = {}

        if st.context.url not in st.session_state.pages:
            st.session_state.pages[st.context.url] = {}

        page = st.session_state.pages[st.context.url]
        # Initialize chat history
        if "messages" not in page:
            page["messages"] = get_system_messages(persona, documents)

        # Display chat messages from history on app rerun
        for message in page["messages"]:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    msg = message["content"]
                    st.markdown(msg)
                    if message["role"] == "assistant":
                        copy_button(msg, key=msg)

        # Await a user message and handle the chat prompt when it comes in.
        if prompt := st.chat_input("Enter a message:"):
            handle_chat_prompt(prompt, page)

    return page


def agent_page_from_key(agent_key: str, header: str, subtitle: str):
    """
    Create an agent page using the agent key to look up the agent's configuration.

    Parameters:
    -----------
    agent_key : str
        The key of the agent to look up in the agent registry.
    header : str
        The header title to display at the top of the page.
    subtitle : str
        The subtitle to display below the header.

    Returns:
    --------
    function
        A page function that can be called to render the chat interface.

    Raises:
    -------
    ValueError
        If the agent key is not found in the registry.
    """
    agent = agent_registry.get(agent_key)
    if not agent:
        raise ValueError(f"Agent '{agent_key}' not found in registry.")

    # Get the persona path
    persona = agent["persona"]

    # Get document(s) - could be a single document or a list
    documents = None
    if "document" in agent:
        documents = agent["document"]
    elif "documents" in agent:
        documents = agent["documents"]

    return agent_page(persona, documents, header, subtitle)


class PageFactory:
    """
    Factory class for creating Streamlit chat pages based on configuration.

    This class allows registering different page types and their corresponding
    creation logic. It supports extensibility by enabling new page types to be
    added dynamically.
    """

    def __init__(self):
        """
        Initialize the PageFactory with an empty registry of page creators.
        """
        self._creators = {}

    def register(self, page_type, creator):
        """
        Register a new page creator for a specific page type.

        Parameters:
        -----------
        page_type : str
            The type or identifier of the page to associate with the creator.
        creator : callable
            A callable or class responsible for creating instances of the specified page type.

        Notes:
        ------
        If a creator is already registered for the given page type, it will be overwritten.
        """
        self._creators[page_type] = creator

    def create(self, page_config):
        """
        Create a page instance based on the provided configuration.

        Parameters:
        -----------
        page_config : dict
            A dictionary containing the configuration for the page.
            Expected keys include:
                - "type" (str, optional): The type of page to create. Defaults to "agent".
                - "agent" (str, optional): The key of the agent to use for the page.
                - "header" (str): The header text for the page.
                - "subtitle" (str): The subtitle text for the page.

        Returns:
        --------
        function
            An instance of the created page.

        Notes:
        ------
        If the specified page type is unknown, a warning is displayed and the default "agent"
        page is created using fallback values.
        """
        page_type = page_config.get("type", "agent")
        creator = self._creators.get(page_type)
        if creator:
            return creator(page_config)
        else:
            st.warning(f"Unknown page type '{page_type}'. Using agent type as default.")
            # Fallback to agent_page with defaults
            return self._creators["agent"](
                {
                    "persona": page_config.get(
                        "persona", "prompts/facilitator_persona.md"
                    ),
                    "document": page_config.get(
                        "document", "prompts/ai_discovery_cards.md"
                    ),
                    "header": page_config["header"],
                    "subtitle": page_config["subtitle"],
                }
            )


# Register page types
page_factory = PageFactory()

# Adapter functions to match the expected signature
page_factory.register(
    "agent",
    lambda cfg: (
        agent_page_from_key(
            cfg["agent"],
            cfg["header"],
            cfg["subtitle"],
        )
        if "agent" in cfg
        else agent_page(
            cfg["persona"],
            cfg.get(
                "documents", cfg.get("document")
            ),  # Try documents first, then document
            cfg["header"],
            cfg["subtitle"],
        )
    ),
)
page_factory.register(
    "multiagent",
    lambda cfg: multiagent_page(
        cfg["personas"],
        cfg["header"],
        cfg["subtitle"],
        cfg.get("documents", None),
    ),
)


def create_page(page_config):
    """
    Factory function to create the appropriate page based on configuration.
    Delegates to PageFactory for extensibility.
    """
    return page_factory.create(page_config)
