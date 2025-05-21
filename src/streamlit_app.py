"""
Main Streamlit application for the Discovery Cards Agent

This is the entry point of the application that handles authentication, page navigation,
and session management. It uses streamlit-authenticator for user authentication and
sets up different persona pages for the AI Discovery Cards workshop.

Key Features:
- User authentication with username/password
- Multi-page navigation with different AI personas
- Session state management for authenticated users
- Configuration loading from YAML files

Dependencies:
- streamlit: Main framework for the web application
- streamlit_authenticator: For user authentication
- yaml: For configuration loading
- openai_page: For creating agent pages with OpenAI integration

Note: The application expects a config.yaml file in the same directory with the proper
authentication configuration format.
"""

import pathlib

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from openai_page import agent_page

with open("./config.yaml", encoding="utf-8") as file:
    config = yaml.load(file, Loader=SafeLoader)

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

st.set_page_config(layout="wide")
# page_icon=str(CURRENT_DIR) + "/assets/logo.png")
# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

try:
    authenticator.login()
    if st.session_state.get("authentication_status"):
        # todo generate pages from a yaml file...
        pages = {
            "Main": [
                st.Page(
                    agent_page(
                        "prompts/facilitator_persona.md",
                        "prompts/ai_discovery_cards.md",
                        "🧑‍🏫 AI Discovery Cards Facilitator",
                        "I'm, your AI Design Thinking Expert and can guide you throuhg the AI Discovery Cards Workshop step by step.",
                    ),
                    title="Facilitator",
                    icon="🧑‍🏫",
                    url_path="Facilitator",
                ),
                st.Page(
                    agent_page(
                        "prompts/customer_persona.md",
                        "prompts/contoso_zermatt_national_bank.md",
                        "🧑‍💼 Contoso Zermatt National Bank Representative",
                        "Ask me anything about our bank, internal processes and our day-to-day jobs.",
                    ),
                    title="Representative",
                    icon="🧑‍💼",
                    url_path="Representative",
                ),
            ]
        }

        pg = st.navigation(pages)
        with st.sidebar:
            authenticator.logout()
        pg.run()
    elif st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")
    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")
    with open("./config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
except (
    yaml.YAMLError,
    FileNotFoundError,
) as e:
    st.error(e)
