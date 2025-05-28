"""
Page components for the Discovery Cards Agent

This module provides page-related classes for building the UI components of the
Discovery Cards Agent application. These classes handle the rendering and interaction
of different types of pages in the Streamlit application.

Classes:
---------
Page:
    Base class for page implementations.

AgentPage:
    Page implementation for a single agent interface.

MultiAgentPage:
    Page implementation for a multi-agent interface.

PageFactory:
    Factory for creating pages based on configuration.
"""

import traceback
from typing import Callable, Dict, List

import streamlit as st
from st_copy import copy_button
from streamlit.logger import get_logger

from agents import MultiAgent, SingleAgent, agent_registry
from utils.openai_utils import handle_chat_prompt, render_message

logger = get_logger(__name__)


class Page:
    """
    Base class for page implementations.

    Attributes:
    -----------
    header : str
        The title displayed at the top of the page.
    subtitle : str
        The subtitle displayed below the header.
    """

    def __init__(self, header: str, subtitle: str) -> None:
        """
        Initialize a Page with header and subtitle.

        Parameters:
        -----------
        header : str
            The title to display at the top of the page.
        subtitle : str
            The subtitle to display below the header.
        """
        self.header = header
        self.subtitle = subtitle

    def render(self) -> None:
        """
        Render the page content in the Streamlit app.

        This base implementation renders the common UI elements like
        the header and subtitle. Subclasses should override this method
        to add specific page content.
        """
        self._add_css()
        st.title(self.header)
        st.subheader(self.subtitle)

    def _add_css(self) -> None:
        """
        Add custom CSS to the page.
        """
        st.write(
            """
            <style>
                div[data-testid="stColumn"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="stColumn"] * {
                    width: fit-content !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def _initialize_session_state(self) -> Dict:
        """
        Initialize the session state for the page.

        Returns:
        --------
        Dict
            The page's session state dictionary.
        """
        if "pages" not in st.session_state:
            st.session_state.pages = {}

        if st.context.url not in st.session_state.pages:
            st.session_state.pages[st.context.url] = {"name": self.header or "Chat"}

        return st.session_state.pages[st.context.url]

    def _reset_chat(self, page: Dict, messages: List[Dict[str, str]]) -> None:
        """
        Reset the chat history.

        Parameters:
        -----------
        page : Dict
            The page's session state.
        messages : List[Dict[str, str]]
            The new system messages to initialize the chat with.
        """
        logger.info("Resetting chat history")
        try:
            page["messages"] = messages
            st.rerun()
        except Exception as e:
            logger.exception("Error resetting chat: %s", e)
            st.error(f"Error resetting chat: {str(e)}")


class AgentPage(Page):
    """
    Page implementation for a single agent interface.

    Attributes:
    -----------
    agent : SingleAgent
        The agent instance that handles the conversation.
    """

    def __init__(self, agent: SingleAgent, header: str, subtitle: str) -> None:
        """
        Initialize an AgentPage with an agent, header, and subtitle.

        Parameters:
        -----------
        agent : SingleAgent
            The agent instance that handles the conversation.
        header : str
            The title to display at the top of the page.
        subtitle : str
            The subtitle to display below the header.
        """
        super().__init__(header, subtitle)
        self.agent = agent

    def render(self) -> None:
        """
        Render the agent page with chat interface.
        """
        super().render()

        # Initialize session state
        page = self._initialize_session_state()

        try:
            # Initialize chat history
            if "messages" not in page:
                logger.debug("Initializing messages for agent %s", self.agent.agent_key)
                page["messages"] = self.agent.get_system_messages()
                logger.debug("Loaded %d system messages", len(page["messages"]))

            msg_count = 0
            # Display chat messages from history on app rerun
            for message in page["messages"]:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        msg = message["content"]
                        render_message(msg)
                        msg_count += 1
                        if message["role"] == "assistant":
                            copy_button(msg, key=msg)

            # Await a user message and handle the chat prompt when it comes in
            if prompt := st.chat_input("Enter a message:"):
                logger.info(
                    "Received user prompt: %s",
                    prompt[:30] + "..." if len(prompt) > 30 else prompt,
                )
                handle_chat_prompt(prompt, page, agent=self.agent)

            # Add control buttons
            self._add_control_buttons(page, msg_count)

        except Exception as e:
            logger.exception("Error rendering agent page: %s", e)
            st.error(f"An error occurred: {str(e)}")
            st.code(traceback.format_exc(), language="python")

    def _add_control_buttons(self, page: Dict, msg_count: int) -> None:
        """
        Add control buttons for rerunning prompts and resetting chat.

        Parameters:
        -----------
        page : Dict
            The page's session state.
        msg_count : int
            The number of messages in the chat history.
        """
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            rerun_prompt_button = st.button(
                "🔄",
                help="Rerun the last prompt",
                use_container_width=True,
            )
            if rerun_prompt_button:
                if msg_count > 0:
                    message = page["messages"][-1]
                    page["messages"].pop()
                    if message["role"] != "user":
                        message = page["messages"][-1]  # Get the last user message
                        page["messages"].pop()

                    handle_chat_prompt(message["content"], page, agent=self.agent)
                else:
                    st.warning("No messages to rerun.")

        with col2:
            reset_button = st.button(
                "🧹", help="Clear chat history", use_container_width=True
            )
            if reset_button:
                self._reset_chat(page, self.agent.get_system_messages())


class MultiAgentPage(Page):
    """
    Page implementation for a multi-agent interface.

    Attributes:
    -----------
    agent : MultiAgent
        The multi-agent instance that handles the conversation.
    """

    def __init__(self, agent: MultiAgent, header: str, subtitle: str) -> None:
        """
        Initialize a MultiAgentPage with a multi-agent, header, and subtitle.

        Parameters:
        -----------
        agent : MultiAgent
            The multi-agent instance that handles the conversation.
        header : str
            The title to display at the top of the page.
        subtitle : str
            The subtitle to display below the header.
        """
        super().__init__(header, subtitle)
        self.agent = agent

    def render(self) -> None:
        """
        Render the multi-agent page with chat interface.
        """
        super().render()

        # Initialize session state
        page = self._initialize_session_state()

        try:
            # Initialize chat history
            if "messages" not in page:
                logger.debug(
                    "Initializing messages for multi-agent %s", self.agent.agent_key
                )
                page["messages"] = self.agent.get_system_messages()
                logger.debug("Loaded %d system messages", len(page["messages"]))

            msg_count = 0
            # Display chat messages from history on app rerun
            for message in page["messages"]:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        msg = message["content"]
                        render_message(msg)
                        msg_count += 1
                        if message["role"] == "assistant":
                            copy_button(msg, key=msg)

            # Await a user message and handle the chat prompt when it comes in.
            if prompt := st.chat_input("Enter a message:"):
                logger.info(
                    "Received user prompt for multi-agent: %s",
                    prompt[:30] + "..." if len(prompt) > 30 else prompt,
                )
                handle_chat_prompt(prompt, page, agent=self.agent)

            # Add control buttons
            self._add_control_buttons(page, msg_count)

        except Exception as e:
            logger.exception("Error rendering multi-agent page: %s", e)
            st.error(f"An error occurred: {str(e)}")
            st.code(traceback.format_exc(), language="python")

    def _add_control_buttons(self, page: Dict, msg_count: int) -> None:
        """
        Add control buttons for rerunning prompts and resetting chat.

        Parameters:
        -----------
        page : Dict
            The page's session state.
        msg_count : int
            The number of messages in the chat history.
        """
        col1, col2 = st.columns([1, 1], gap="small")
        with col1:
            rerun_prompt_button = st.button(
                "🔄",
                help="Rerun the last prompt",
                use_container_width=True,
            )
            if rerun_prompt_button and msg_count > 0:
                message = page["messages"][-1]
                page["messages"].pop()
                if message["role"] != "user":
                    message = page["messages"][-1]  # Get the last user message
                    page["messages"].pop()

                handle_chat_prompt(message["content"], page, agent=self.agent)
            elif rerun_prompt_button and msg_count == 0:
                st.warning("No messages to rerun.")

        with col2:
            reset_button = st.button(
                "🧹", help="Clear chat history", use_container_width=True
            )
            if reset_button:
                self._reset_chat(page, self.agent.get_system_messages())


class PageFactory:
    """
    Factory for creating pages based on configuration.
    """

    @staticmethod
    def create_page(page_config: Dict) -> Callable[[], None]:
        """
        Create a page function based on page configuration.

        Parameters:
        -----------
        page_config : Dict
            Configuration for the page.

        Returns:
        --------
        Callable[[], None]
            A function that renders the page.
        """
        try:
            page_type = page_config.get("type", "agent")
            header = page_config.get("header", "")
            subtitle = page_config.get("subtitle", "")

            logger.info(
                "Creating page of type '%s' with header '%s'", page_type, header
            )

            # Check if we're using agent_key or direct configuration
            if "agent" in page_config:
                # Using agent key from registry
                agent_key = page_config["agent"]
                logger.debug("Using agent '%s' from registry", agent_key)

                agent = agent_registry.get_agent(agent_key)
                if not agent:
                    error_msg = f"Agent '{agent_key}' not found in registry"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                if isinstance(agent, MultiAgent):
                    page = MultiAgentPage(agent, header, subtitle)
                elif isinstance(agent, SingleAgent):
                    page = AgentPage(agent, header, subtitle)
                else:
                    error_msg = f"Unsupported agent type: {type(agent)}"
                    logger.error(error_msg)
                    raise TypeError(error_msg)

            elif page_type == "multiagent":
                # Direct multiagent configuration
                personas = page_config["personas"]
                model = page_config.get("model", "gpt-4o")
                documents = page_config.get("documents")
                temperature = page_config.get("temperature", 1)

                logger.debug(
                    "Creating custom MultiAgent with %d personas", len(personas)
                )

                agent = MultiAgent(
                    agent_key="_custom_",  # Using a placeholder key
                    personas=personas,
                    model=model,
                    documents=documents,
                    temperature=temperature,
                )
                page = MultiAgentPage(agent, header, subtitle)

            else:
                # Default to single agent configuration
                persona = page_config.get("persona", "prompts/facilitator_persona.md")
                model = page_config.get("model", "gpt-4o")
                document = page_config.get("document")
                documents = page_config.get("documents")
                temperature = page_config.get("temperature", 1)

                # Determine document source
                doc_source = None
                if documents:
                    doc_source = documents
                elif document:
                    doc_source = document

                logger.debug("Creating custom SingleAgent with persona %s", persona)

                agent = SingleAgent(
                    agent_key="_custom_",  # Using a placeholder key
                    persona=persona,
                    model=model,
                    documents=doc_source,
                    temperature=temperature,
                )
                page = AgentPage(agent, header, subtitle)

            # Return a function that renders the page
            return page.render

        except Exception as e:
            logger.exception("Error creating page: %s", e)
            error_msg = f"Error creating page: {str(e)}"
            raise ValueError(error_msg) from e
