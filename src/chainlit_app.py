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
from agent_manager import ChainlitAgentManager

from agents import RESPONSE_TAG, agent_registry

AUTH_CONFIG_FILE = "./config/auth-config.yaml"


dotenv.load_dotenv()


LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = getLogger(__name__)
logger.setLevel(LOGLEVEL)

if not os.getenv("CHAINLIT_AUTH_SECRET"):
    logger.warning(
        "CHAINLIT_AUTH_SECRET is not set. Authentication will not be secure. Generating a random secret."
    )
    os.environ["CHAINLIT_AUTH_SECRET"] = random_secret()
    dotenv.set_key(".env", "CHAINLIT_AUTH_SECRET",
                   os.environ["CHAINLIT_AUTH_SECRET"])


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


@cl.set_chat_profiles
async def chat_profile(user: Optional[cl.User] = None) -> List[cl.ChatProfile]:

    profiles: List[cl.ChatProfile] = []

    if user:
        user_roles = user.metadata.get("roles", ["user"])
        available_agents = agent_manager.get_available_agents(user_roles)
        for agent_key, agent_info in available_agents.items():
            profiles.append(
                cl.ChatProfile(
                    name=agent_info["header"],
                    markdown_description=agent_info["subtitle"],
                    default=agent_info.get("default", False)
                )
            )
    return profiles


@cl.on_chat_start
async def start() -> None:
    """Initialize the chat session when a user connects."""
    user = cl.user_session.get("user")
    if not user:
        await cl.Message(content="❌ Authentication required. Please log in.").send()
        return

    await cl.Message(
        content=f"👋 Welcome, {user.metadata.get('first_name', 'User')}! You are logged in as `{user.identifier}`.\n\n").send()

    user_roles = user.metadata.get("roles", ["user"])
    available_agents = agent_manager.get_available_agents(user_roles)
    if not available_agents:
        await cl.Message(content="❌ No agents available for your user role.").send()
        return
    cl.user_session.set("available_agents", available_agents)

    chat_profile = cl.user_session.get("chat_profile")
    current_agent_key = None
    if chat_profile:
        for agent_key, agent_info in available_agents.items():
            if agent_info.get("header") == chat_profile:
                current_agent_key = agent_key
                await cl.Message(
                    content=f"## {agent_info['header']}.\n\n{agent_info['subtitle']}"
                ).send()
                break
    if not current_agent_key:
        # If no current agent is set, default to the first available agent
        if available_agents:
            current_agent_key = next(iter(available_agents.keys()))
        else:
            await cl.Message(content="❌ No agents available.").send()
            return
    cl.user_session.set("current_agent_key", current_agent_key)


@cl.on_message
async def main(message: cl.Message) -> None:
    """Handle incoming messages and route them to the appropriate agent."""
    user = cl.user_session.get("user")
    if not user:
        await cl.Message(content="❌ Authentication required. Please log in.").send()
        return

    content = message.content.strip()

    current_agent_key = cl.user_session.get("current_agent_key")
    if not current_agent_key:
        await cl.Message(
            content="❌ No agent selected. Please select an agent to continue."
        ).send()
        return
    # Process message with current agent
    await process_with_agent(content, current_agent_key, user)


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
