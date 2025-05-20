import streamlit as st
from time import sleep
from streamlit.runtime.scriptrunner import get_script_run_ctx
import pathlib

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

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
    st.logo(str(CURRENT_DIR) + "/assets/logo.png", size='medium')
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.subheader("🗂️ AI Discovery Cards")
        st.write("")
        st.write("")

        if st.session_state.get('authentication_status'):
            st.page_link("pages/1_Facilitator.py", label="Facilitator", icon="🧑‍🏫")
            st.page_link("pages/2_Customer.py", label="Representative", icon="🧑‍💼")

            st.write("")
            st.write("")

            authenticator.logout()

        elif get_current_page_name() != "streamlit app":
            # If anyone tries to access a secret page without being logged in,
            # redirect them to the login page
            st.switch_page("streamlit_app.py")
