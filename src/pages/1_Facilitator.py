import streamlit as st
from utils.openai_utils import handle_chat_prompt, load_prompt_files

from navigation import make_sidebar

make_sidebar()

@st.cache_data
def get_init_messages():
    """Return the initial messages for the chat history."""
    return load_prompt_files(
        "prompts/facilitator_persona.md",
        "prompts/ai_discovery_cards.md"
    )


st.write(
"""
# 🧑‍🏫 AI Discovery Cards Facilitator
"""
)

if "pages" not in st.session_state:
    st.session_state.pages = {} 

if st.context.url not in st.session_state.pages:
    st.session_state.pages[st.context.url] = {}

page=st.session_state.pages[st.context.url]
# Initialize chat history
if "messages" not in page:
    page["messages"] = get_init_messages()


# Display chat messages from history on app rerun
for message in page["messages"]:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Await a user message and handle the chat prompt when it comes in.
if prompt := st.chat_input("Enter a message:"):
    handle_chat_prompt(prompt, page)

    