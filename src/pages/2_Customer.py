# code from https://docs.streamlit.io/library/get-started/create-an-app

import streamlit as st
import openai
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from streamlit.logger import get_logger
import re

from dotenv import load_dotenv
import os

# Configure logging using Streamlit's logger
logger = get_logger(__name__)

# Ensure the logger has at least one handler


load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

st.set_page_config(layout="wide")

def create_chat_completion(messages):
    """Create and return a new chat completion request. Key assumptions:
    - The Azure OpenAI endpoint and deployment name are stored in Streamlit secrets."""

    # Retrieve secrets from the Streamlit secret store.
    # This is a secure way to store sensitive information that you don't want to expose in your code.
    # Learn more about Streamlit secrets here: https://docs.streamlit.io/develop/concepts/connections/secrets-management
    # The secrets themselves are stored in the .streamlit/secrets.toml file.

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    
    aoai_endpoint = AZURE_OPENAI_ENDPOINT
    aoai_deployment_name = "gpt-4o"

    # search_endpoint = st.secrets["search"]["endpoint"]
    # search_key = st.secrets["search"]["key"]
    # search_index_name = st.secrets["search"]["index_name"]


    client = openai.AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2024-06-01",
        azure_endpoint = aoai_endpoint
    )
    # Create and return a new chat completion request
    return client.chat.completions.create(
          model=aoai_deployment_name,
          messages=[
              {"role": m["role"], "content": m["content"]}
              for m in messages
          ],
          stream=True,
          stream_options={
              "include_usage": True
          }
            #,
        #   extra_body={
        #       "data_sources": [
        #           {
        #               "type": "azure_search",
        #               "parameters": {
        #                   "endpoint": search_endpoint,
        #                   "index_name": search_index_name,
        #                   "authentication": {
        #                       "type": "api_key",
        #                       "key": search_key
        #                   }
        #               }
        #           }
        #       ]
        #   }
      )


def count_xml_tags(text):
    # Define the regex pattern for XML tags
    pattern = r'<[^>]+>'

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Return the number of matches
    return len(matches)


def handle_chat_prompt(prompt):
    """Echo the user's prompt to the chat window.
    Then, send the user's prompt to Azure OpenAI and display the response."""

    # Cleanup prompt
    if count_xml_tags(prompt)>0:
        logger.debug("Prompt contains XML tags.")
        # embed documents to avoid harm
        prompt=f"<documents>{prompt}</documents>"

    # Echo the user's prompt to the chat window
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    # Calculate tokens in the input
    input_tokens = count_tokens(st.session_state.messages)
    
    # Send the user's prompt to Azure OpenAI and display the response
    # The call to Azure OpenAI is handled in create_chat_completion()
    # This function loops through the responses and displays them as they come in.
    # It also appends the full response to the chat history.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        completion = None
        for response in create_chat_completion(st.session_state.messages):
            if response.choices:
                try:
                    if response.choices[0].delta != None:
                        full_response += (response.choices[0].delta.content or "")
                        message_placeholder.markdown(full_response + "▌")
                    else:
                        logger.debug(response.choices[0].model_dump_json())
                except Exception as e:
                    logger.exception(e)
                    full_response += "An error happened, retry your request.\n"
            completion = response
        message_placeholder.markdown(full_response)
    
    # Add the response to the messages
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Display token usage
    if completion and completion.usage:
        st.caption(f"""Token usage for this interaction:
        - Input tokens: {input_tokens}
        - Output tokens: {completion.usage.completion_tokens}
        - Total tokens: {completion.usage.total_tokens}""")
    

def get_init_messages():
    """Return the initial messages for the chat history.
    This is a simple system message that sets the context for the chat."""

    # Read the files facilitator_persona.md and ai_discovery_cards.md into a list of strings
    with open("prompts/customer_persona.md", "r") as f:
        facilitator_persona = f.read()
    with open("prompts/contoso_zermatt_national_bank.md", "r") as f:
        ai_discovery_cards = f.read()
    return [
        {
            "role": "system",
            "content": facilitator_persona
        },
        {
            "role": "system",
            "content": f"Find the use case below:\n\n{ai_discovery_cards}"
        }
    ]

def count_tokens(messages):
    """Count the number of tokens in the messages."""
    encoding = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by GPT-4
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens

def main():
    """Main function for the Chat with Data Streamlit app."""

    st.write(
    """
    # Chat with Data

    This Streamlit dashboard is intended to show off capabilities of Azure OpenAI.
    """
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = get_init_messages()


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Await a user message and handle the chat prompt when it comes in.
    if prompt := st.chat_input("Enter a message:"):
        handle_chat_prompt(prompt)

if __name__ == "__main__":
    main()