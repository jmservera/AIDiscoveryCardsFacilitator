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

from typing import Any

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from openai_page import create_page


def initialize_authentication():
    """
    Initializes user authentication for the Streamlit app using configuration from a YAML file.

    This function loads authentication and cookie settings from a local 'config.yaml' file,
    sets the Streamlit page layout, and returns an instance of the stauth.Authenticate class
    configured with the loaded credentials and cookie parameters.

    Returns:
        stauth.Authenticate: An authentication object configured with credentials and cookie settings.
    """
    with open("./config.yaml", encoding="utf-8") as file:
        configuration = yaml.load(file, Loader=SafeLoader)

    st.set_page_config(layout="wide")
    # Pre-hashing all plain text passwords once
    # stauth.Hasher.hash_passwords(config['credentials'])

    return (
        stauth.Authenticate(
            configuration["credentials"],
            configuration["cookie"]["name"],
            configuration["cookie"]["key"],
            configuration["cookie"]["expiry_days"],
        ),
        configuration,
    )


authenticator, config = initialize_authentication()


def clear(_: Any) -> None:
    """Clear the session state and redirect to the login page."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def login_page() -> None:
    """Render the login page."""
    authenticator.login()

    if st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect")
    elif st.session_state.get("authentication_status") is None:
        st.warning("Please enter your username and password")


def main() -> None:
    """Main function to run the Streamlit application."""
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
                    is_new_section = True
                    for page_config in section_pages:
                        # Hide admin_only pages for non-admins
                        if page_config.get("admin_only", False) and not is_admin:
                            continue
                        if is_new_section:
                            pages[section] = []
                            is_new_section = False
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


if __name__ == "__main__":
    # Run the main function
    main()
