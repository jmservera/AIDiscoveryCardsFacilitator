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
- Dynamic page generation from pages.yaml configuration

Dependencies:
- streamlit: Main framework for the web application
- streamlit_authenticator: For user authentication
- yaml: For configuration loading
- openai_page: For creating agent pages with OpenAI integration

Configuration Files:
- config.yaml: Authentication configuration
- pages.yaml: Page structure and content configuration

Note: The application expects the configuration files in the same directory with the proper
format.
"""

import pathlib

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from openai_page import agent_page, create_page

with open("./config.yaml", encoding="utf-8") as file:
    config = yaml.load(file, Loader=SafeLoader)

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

st.set_page_config(layout="wide")
# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)


def clear(values):
    """Clear the session state and redirect to the login page."""
    for key in st.session_state.keys():
        del st.session_state[key]


def login_page():
    """Render the login page."""
    authenticator.login()

    if st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")
    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")


try:
    if st.session_state.get("authentication_status"):
        # Load pages from pages.yaml file
        try:
            with open("./pages.yaml", encoding="utf-8") as file:
                pages_config = yaml.load(file, Loader=SafeLoader)

            # Determine if the user is an admin
            user_roles = st.session_state.get("roles")
            is_admin = "admin" in user_roles

            # Convert YAML configuration to streamlit pages structure
            pages = {}
            for section, section_pages in pages_config["sections"].items():
                pages[section] = []
                for page_config in section_pages:
                    # Hide admin_only pages for non-admins
                    if page_config.get("admin_only", False) and not is_admin:
                        continue
                    page_func = create_page(page_config)
                    page = st.Page(
                        page_func,
                        title=page_config["title"],
                        icon=page_config["icon"],
                        url_path=page_config["url_path"],
                    )
                    pages[section].append(page)

            pg = st.navigation(pages, position="sidebar")
            authenticator.logout(location="sidebar", callback=clear)
            pg.run()
        except (yaml.YAMLError, FileNotFoundError) as page_error:
            st.error(f"Error loading pages configuration: {page_error}")

    else:
        pg = st.navigation([login_page], position="hidden")
        pg.run()
    with open("./config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
except (
    yaml.YAMLError,
    FileNotFoundError,
) as e:
    st.error(e)
