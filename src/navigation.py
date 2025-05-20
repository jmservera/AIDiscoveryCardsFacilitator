import streamlit as st
from time import sleep
from streamlit.runtime.scriptrunner import get_script_run_ctx

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

def get_current_page_name():
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Couldn't get script context")

    pages = ctx.pages_manager.get_pages()

    return pages[ctx.page_script_hash]["page_name"]


def make_sidebar():
    with st.sidebar:
        st.title("💎 AI Design Cards Facilitator")
        st.write("")
        st.write("")

        if st.session_state.get('authentication_status'):
            st.page_link("pages/1_Facilitator.py", label="Facilitator", icon="🔒")
            st.page_link("pages/2_Customer.py", label="Contoso Representative", icon="🕵️")

            st.write("")
            st.write("")

            authenticator.logout()

        elif get_current_page_name() != "streamlit app":
            # If anyone tries to access a secret page without being logged in,
            # redirect them to the login page
            st.switch_page("streamlit_app.py")
