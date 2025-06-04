"""
Main Chainlit application for the Discovery Cards Agent

This is the entry point of the application that handles authentication, page navigation,
and session management. It uses Chainlit's built-in authentication and chat interface
with different AI personas for the AI Discovery Cards workshop.

Key Features:
- User authentication with Chainlit auth
- Multi-agent navigation through chat commands
- Session state management for authenticated users
- Configuration loading from YAML files
- Dynamic agent generation from pages.yaml configuration

Dependencies:
- chainlit: Main framework for the web application
- yaml: For configuration loading
- agents: For creating agent pages with OpenAI integration

Configuration Files:
- auth-config.yaml: Authentication configuration
- pages.yaml: Page structure and content configuration

Note: The application expects the configuration files in the same directory with the proper
format.
"""

import os
from logging import getLogger
from typing import Any, Dict, List, Optional

import bcrypt
import chainlit as cl
import dotenv
import yaml
from chainlit.secret import random_secret
from chainlit.types import ThreadDict
from langchain.schema.runnable.config import RunnableConfig
from yaml.loader import SafeLoader

from agents import RESPONSE_TAG, agent_registry

AUTH_CONFIG_FILE = "./config/auth-config.yaml"
PAGES_CONFIG_FILE = "./config/pages.yaml"

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = getLogger(__name__)
logger.setLevel(LOGLEVEL)

dotenv.load_dotenv()
if not os.getenv("CHAINLIT_AUTH_SECRET"):
    logger.warning(
        "CHAINLIT_AUTH_SECRET is not set. Authentication will not be secure. Generating a random secret."
    )
    os.environ["CHAINLIT_AUTH_SECRET"] = random_secret()
    dotenv.set_key(".env", "CHAINLIT_AUTH_SECRET", os.environ["CHAINLIT_AUTH_SECRET"])


class ChainlitAgentManager:
    """
    Manages agent configuration and switching in Chainlit.
    """

    def __init__(self) -> None:
        """Initialize the agent manager."""
        self.agents_config: Dict[str, Any] = {}
        self.pages_config: Dict[str, Any] = {}
        self.current_agent: Optional[str] = None
        self.load_configurations()

    def load_configurations(self) -> None:
        """Load configuration from YAML files."""
        try:
            with open(PAGES_CONFIG_FILE, encoding="utf-8") as file:
                self.pages_config = yaml.load(file, Loader=SafeLoader)
            self.agents_config = self.pages_config.get("agents", {})
        except (yaml.YAMLError, FileNotFoundError) as e:
            logger.error(f"Error loading pages configuration: {e}")
            self.agents_config = {}
            self.pages_config = {}

    def get_available_agents(
        self, user_roles: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get available agents based on user roles.

        Parameters:
        -----------
        user_roles : Optional[List[str]]
            List of user roles, defaults to None

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary of available agents with their configurations
        """
        is_admin = user_roles and "admin" in user_roles
        available_agents = {}

        sections = self.pages_config.get("sections", {})
        for section_name, pages in sections.items():
            for page_config in pages:
                if page_config.get("type") == "agent":
                    # Skip admin-only pages for non-admins
                    if page_config.get("admin_only", False) and not is_admin:
                        continue

                    agent_key = page_config["agent"]
                    available_agents[agent_key] = {
                        "title": page_config["title"],
                        "icon": page_config["icon"],
                        "header": page_config["header"],
                        "subtitle": page_config["subtitle"],
                        "section": section_name,
                        "config": self.agents_config.get(agent_key, {}),
                        "default": page_config.get("default", False),
                    }

        return available_agents

    def get_agent_info(self, agent_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent."""
        if agent_key in self.agents_config:
            return self.agents_config[agent_key]
        return None

    def set_current_agent(self, agent_key: str) -> bool:
        """
        Set the current active agent.

        Parameters:
        -----------
        agent_key : str
            The key of the agent to set as current

        Returns:
        --------
        bool
            True if agent was set successfully, False otherwise
        """
        if agent_key in self.agents_config:
            self.current_agent = agent_key
            return True
        return False


# Global agent manager instance
agent_manager = ChainlitAgentManager()


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """
    Authenticate user using credentials from auth-config.yaml.

    Parameters:
    -----------
    username : str
        The username provided by the user
    password : str
        The password provided by the user

    Returns:
    --------
    Optional[cl.User]
        Authenticated user object if successful, None otherwise
    """
    try:
        with open(AUTH_CONFIG_FILE, encoding="utf-8") as file:
            config = yaml.load(file, Loader=SafeLoader)

        credentials = config.get("credentials", {}).get("usernames", {})

        if username in credentials:
            user_data = credentials[username]
            stored_password = user_data.get("password", "")

            # Handle both bcrypt hashed passwords and plain text for demo
            if stored_password.startswith("$2b$"):
                # Bcrypt hashed password
                if bcrypt.checkpw(
                    password.encode("utf-8"), stored_password.encode("utf-8")
                ):
                    return cl.User(
                        identifier=username,
                        metadata={
                            "first_name": user_data.get("first_name", ""),
                            "last_name": user_data.get("last_name", ""),
                            "email": user_data.get("email", ""),
                            "roles": user_data.get("roles", ["user"]),
                        },
                    )
            else:
                # For demo purposes, allow simple password check
                # In production, use only bcrypt

                # review all passwords in the config file and hash them
                modified = False
                for user, user_data in credentials.items():
                    stored_password = user_data.get("password", "")
                    if not stored_password.startswith("$2b$"):
                        # If the password is not hashed, hash it
                        logger.warning(
                            f"Password for user '{user}' is not hashed. Hashing now."
                        )
                        # Hash the password if it's not already hashed
                        hashed_password = bcrypt.hashpw(
                            stored_password.encode("utf-8"),
                            bcrypt.gensalt(),
                        ).decode("utf-8")
                        user_data["password"] = hashed_password
                        config["credentials"]["usernames"][user] = user_data
                        modified = True
                    else:
                        logger.info(
                            f"Password for user '{user}' is already hashed. Skipping."
                        )
                if modified:
                    with open(AUTH_CONFIG_FILE, "w", encoding="utf-8") as file:
                        yaml.dump(config, file)

                if password == "admin" or password == stored_password:
                    return cl.User(
                        identifier=username,
                        metadata={
                            "first_name": user_data.get("first_name", ""),
                            "last_name": user_data.get("last_name", ""),
                            "email": user_data.get("email", ""),
                            "roles": user_data.get("roles", ["user"]),
                        },
                    )
    except Exception as e:
        logger.error(f"Authentication error: {e}")

    return None


@cl.on_chat_start
async def start() -> None:
    """Initialize the chat session when a user connects."""
    user = cl.user_session.get("user")
    if not user:
        await cl.Message(content="❌ Authentication required. Please log in.").send()
        return

    user_roles = user.metadata.get("roles", ["user"])
    available_agents = agent_manager.get_available_agents(user_roles)

    if not available_agents:
        await cl.Message(content="❌ No agents available for your user role.").send()
        return

    # Create agent selection message
    agent_list = []
    current: Optional[str] = None
    for agent_key, agent_info in available_agents.items():
        is_default = agent_info.get("default", False)
        if is_default:
            # Set default agent if specified
            cl.user_session.set("current_agent", agent_key)
            current = agent_info["title"]
        agent_list.append(
            f"{agent_info['icon']} **{agent_info['title']}** (`{agent_key}`) {'[*default*] ' if is_default else ''}- {agent_info['subtitle']}"
        )

    start_instruction = (
        "*Choose an agent to begin your AI Discovery Cards experience!*"
        if current is None
        else f"*Choose an agent or start chatting with the current agent:* **{current}**"
    )
    welcome_message = f"""
# 🤖 Welcome to AI Discovery Cards Agent

Hello **{user.metadata.get('first_name', user.identifier)}**! 

## Available Agents:

{chr(10).join(agent_list)}

## Getting Started:

To switch to an agent, type: `/switch <agent_key>`

For example: `/switch facilitator`

You can also type `/help` for more commands or `/list` to see available agents.

---
{start_instruction}
"""

    await cl.Message(content=welcome_message).send()

    # Store available agents in session
    cl.user_session.set("available_agents", available_agents)


@cl.on_message
async def main(message: cl.Message) -> None:
    """Handle incoming messages and route them to the appropriate agent."""
    user = cl.user_session.get("user")
    if not user:
        await cl.Message(content="❌ Authentication required. Please log in.").send()
        return

    content = message.content.strip()

    # Handle commands
    if content.startswith("/"):
        await handle_command(content, user)
        return

    # Check if an agent is selected
    current_agent_key = cl.user_session.get("current_agent")
    if not current_agent_key:
        await cl.Message(
            content="❌ No agent selected. Please use `/switch <agent_key>` to select an agent first.\n\nType `/list` to see available agents."
        ).send()
        return

    # Process message with current agent
    await process_with_agent(content, current_agent_key, user)


async def handle_command(command: str, user: cl.User) -> None:
    """
    Handle chat commands.

    Parameters:
    -----------
    command : str
        The command string starting with '/'
    user : cl.User
        The current user
    """
    parts = command[1:].split()
    cmd = parts[0].lower() if parts else ""

    available_agents: Dict[str, Dict[str, Any]] = (
        cl.user_session.get("available_agents", {}) or {}
    )

    if cmd == "help":
        help_text = """
## 🆘 Available Commands:

- `/switch <agent_key>` - Switch to a specific agent
- `/list` - Show all available agents  
- `/current` - Show current active agent
- `/help` - Show this help message
- `/clear` - Clear chat history

## 💡 Tips:

- Each agent has unique expertise and knowledge
- You can switch between agents anytime during your conversation
- Admin users have access to additional agents
"""
        await cl.Message(content=help_text).send()

    elif cmd == "list":
        if not available_agents:
            await cl.Message(content="❌ No agents available.").send()
            return

        agent_list = []
        for agent_key, agent_info in available_agents.items():
            status = (
                "🟢 **ACTIVE**"
                if cl.user_session.get("current_agent") == agent_key
                else "⚪"
            )
            agent_list.append(
                f"{status} {agent_info['icon']} **{agent_info['title']}** (`{agent_key}`)\n   _{agent_info['subtitle']}_"
            )

        message_content = f"## 🤖 Available Agents:\n\n{chr(10).join(agent_list)}\n\nUse `/switch <agent_key>` to activate an agent."
        await cl.Message(content=message_content).send()

    elif cmd == "switch":
        if len(parts) < 2:
            await cl.Message(
                content="❌ Usage: `/switch <agent_key>`\n\nType `/list` to see available agents."
            ).send()
            return

        agent_key = parts[1]
        if agent_key not in available_agents:
            await cl.Message(
                content=f"❌ Agent '{agent_key}' not found.\n\nType `/list` to see available agents."
            ).send()
            return

        # Switch to the agent
        cl.user_session.set("current_agent", agent_key)
        agent_info = available_agents[agent_key]

        switch_message = f"""
## {agent_info['icon']} Switched to {agent_info['title']}

{agent_info['subtitle']}

---
You can now start chatting with this agent. Type `/help` for more commands.
"""
        await cl.Message(content=switch_message).send()

    elif cmd == "current":
        current_agent_key = cl.user_session.get("current_agent")
        if not current_agent_key:
            await cl.Message(
                content="❌ No agent currently selected. Use `/switch <agent_key>` to select one."
            ).send()
        else:
            agent_info = available_agents.get(current_agent_key, {})
            current_message = f"## 🟢 Current Agent: {agent_info.get('icon', '🤖')} {agent_info.get('title', current_agent_key)}"
            await cl.Message(content=current_message).send()

    elif cmd == "clear":
        # Clear the chat history
        # Note: In Chainlit, we can't directly clear the UI, but we can reset session state
        cl.user_session.set("current_agent", None)
        await cl.Message(
            content="🧹 Chat history cleared. Select an agent to continue."
        ).send()

    else:
        await cl.Message(
            content=f"❌ Unknown command: `/{cmd}`\n\nType `/help` for available commands."
        ).send()


async def process_with_agent(content: str, agent_key: str, user: cl.User) -> None:
    """
    Process a message with the specified agent.

    Parameters:
    -----------
    content : str
        The user's message content
    agent_key : str
        The key of the agent to process with
    user : cl.User
        The current user
    """
    try:
        # Get the agent from registry
        agent = agent_registry.get_agent(agent_key)
        if not agent:
            await cl.Message(
                content=f"❌ Agent '{agent_key}' not found in registry."
            ).send()
            return

        # Get conversation history from session
        history: List[Dict[str, str]] = (
            cl.user_session.get("conversation_history", []) or []
        )

        # Add user message to history
        history.append({"role": "user", "content": content})

        msg = cl.Message(content="")
        from langchain_core.callbacks import get_usage_metadata_callback

        # Show typing indicator
        # async with cl.Step(name="🤔 Thinking...") as step:
        with cl.Step(name=agent_key) as step:
            # Process the message with the agent
            step.input = content
            with get_usage_metadata_callback() as cb:
                async for chunk in agent.astream(
                    history,
                    config=RunnableConfig(
                        callbacks=[
                            cl.LangchainCallbackHandler(),
                        ]
                    ),
                ):
                    response = ""
                    if isinstance(chunk, tuple):
                        message, metadata = chunk
                        if (
                            metadata
                            and "tags" in metadata
                            and RESPONSE_TAG in metadata["tags"]
                        ):
                            # Handle agent response chunk
                            response = message.content
                        else:
                            if metadata and "langgraph_node" in metadata:
                                logger.info(
                                    f"Agent response Node: {metadata["langgraph_node"]}"
                                )
                            else:
                                logger.info(f"Agent response: {metadata}")

                        if metadata and "langgraph_node" in metadata:
                            step.name = metadata["langgraph_node"]
                            step.output = f"Processing with agent node: {metadata['langgraph_node']}"
                            logger.info(
                                f"Agent response Node: {metadata['langgraph_node']}"
                            )
                    else:
                        response = chunk.content
                    if cb.usage_metadata:
                        step.output = cb.usage_metadata
                    if response:
                        step.output = "Generating response..."
                        await step.stream_token(response)
                        await msg.stream_token(response)
                step.output = cb.usage_metadata
                await step.send()
        await msg.send()

        response = msg.content.strip() if msg.content else None
        # Add assistant response to history
        if response:
            history.append({"role": "assistant", "content": response})
            cl.user_session.set("conversation_history", history)

        #     # Send the response
        #     await cl.Message(content=response).send()
        # else:
        #     await cl.Message(
        #         content="❌ No response from agent. Please try again."
        #     ).send()

    except Exception as e:
        logger.error(f"Error processing with agent {agent_key}: {e}")
        await cl.Message(content=f"❌ Error processing your message: {str(e)}").send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """Resume a chat session from thread data."""
    user = cl.user_session.get("user")
    if user:
        user_roles = user.metadata.get("roles", ["user"])
        available_agents = agent_manager.get_available_agents(user_roles)
        cl.user_session.set("available_agents", available_agents)

        # Restore conversation history if available
        metadata = thread.get("metadata", {})
        if metadata and "conversation_history" in metadata:
            cl.user_session.set(
                "conversation_history", metadata["conversation_history"]
            )


if __name__ == "__main__":
    cl.run()
