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

from utils.openai_utils import handle_chat_prompt, load_prompt_files


@st.cache_data
def get_system_messages(persona: str, document: str):
    """Return the initial messages for the chat history."""
    return load_prompt_files(persona, document)


@st.cache_data
def get_system_messages_multiagent(personas: list[str], documents: list[str] = None):
    """Return the initial messages for a multi-agent chat history.

    If documents are not provided, only persona prompts will be used.
    """
    messages = []
    for i, persona in enumerate(personas):
        doc = documents[i] if documents and i < len(documents) else ""
        persona_messages = load_prompt_files(persona, doc)
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


def agent_page(persona: str, document: str, header: str, subtitle: str):
    """
    Create a Streamlit chat page for interacting with an agent persona.

    This function returns a page function that sets up a chatbot interface with a specific
    persona and document context. The page includes a header, subtitle, and a chat history
    that persists across page reloads by storing it in the Streamlit session state.

    Parameters:
    -----------
    persona : str
        The persona or role prompt file name that the assistant should adopt in the conversation.
    document : str
        The document or context file that the assistant should use as reference.
    title : str
        The title to display at the top of the page.
    subtitle : str
        The subtitle to display below the title.

    Returns:
    --------
    function
        A page function that can be called to render the chat interface.

    Notes:
    ------
    The function relies on several external dependencies:
    - get_system_messages(): From the openai_utils module.
                             Expected to initialize the chat with system messages
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
            page["messages"] = get_system_messages(persona, document)

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


class PageFactory:
    """
    Factory class for creating Streamlit chat pages based on configuration.
    """

    def __init__(self):
        self._creators = {}

    def register(self, page_type, creator):
        """
        Registers a new page creator for a specific page type.

        Args:
            page_type: The type or identifier of the page to associate with the creator.
            creator: A callable or class responsible for creating instances of the specified page type.

        Raises:
            None explicitly, but may overwrite an existing creator for the given page_type.
        """
        self._creators[page_type] = creator

    def create(self, page_config):
        """
        Creates a page instance based on the provided configuration.

        Args:
            page_config (dict): A dictionary containing the configuration for the page.
                Expected keys include:
                    - "type" (str, optional): The type of page to create. Defaults to "agent".
                    - "persona" (str, optional): Path to the persona file. Used if type is "agent".
                    - "document" (str, optional): Path to the document file. Used if type is "agent".
                    - "header" (str): The header text for the page.
                    - "subtitle" (str): The subtitle text for the page.

        Returns:
            object: An instance of the created page.

        Notes:
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
    lambda cfg: agent_page(
        cfg["persona"],
        cfg["document"],
        cfg["header"],
        cfg["subtitle"],
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
