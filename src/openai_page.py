"""
Streamlit page components for OpenAI-powered chat interfaces

This module provides components and utilities for building interactive chat interfaces that
communicate with Azure OpenAI. It handles the UI/UX aspects of the chat experience and
delegates the AI interactions to the openai_utils module.

Key Features:
- Cached system message initialization
- Dynamic chat page generation with customizable personas
- Persistent chat history across page reloads
- Integration with OpenAI's chat completion API

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


def agent_page(persona: str, document: str, title: str, subtitle: str):
    """
    Create a Streamlit chat page for interacting with an agent persona.

    This function returns a page function that sets up a chatbot interface with a specific
    persona and document context. The page includes a title, subtitle, and a chat history
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
        st.title(title)
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
