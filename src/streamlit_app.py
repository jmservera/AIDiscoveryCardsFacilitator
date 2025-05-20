import streamlit as st
from time import sleep
from navigation import make_sidebar

import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader

make_sidebar()

st.title("Welcome to The AI Discovery Cards Workshop")

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

try:
    authenticator.login()
    if st.session_state.get('authentication_status'):
        st.write(f'Welcome *{st.session_state.get("name")}*')
        sleep(0.5)
        st.switch_page("pages/1_Facilitator.py")
    elif st.session_state.get('authentication_status') is False:
            st.error('Username/password is incorrect')
    elif st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')
    with open('./config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
except Exception as e:
    st.error(e)
