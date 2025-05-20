"""
Utility functions for interacting with Azure OpenAI API
"""

import streamlit as st
import openai
import tiktoken
import re
import os
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from streamlit.logger import get_logger

# Configure logging using Streamlit's logger
logger = get_logger(__name__)

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

@st.cache_resource
def get_client() -> openai.AzureOpenAI:
    """Get the Azure OpenAI client using DefaultAzureCredential.
    
    Returns:
        An authenticated Azure OpenAI client
    """
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    
    return openai.AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2024-06-01",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

def create_chat_completion(messages):
    """Create and return a new chat completion request.
    
    Args:
        messages: List of message objects with role and content
        azure_openai_endpoint: Azure OpenAI endpoint URL
    
    Returns:
        A streaming response from Azure OpenAI
    """

    client = get_client()
    
    # Create and return a new chat completion request
    return client.chat.completions.create(
          model=AZURE_OPENAI_DEPLOYMENT_NAME,
          messages=[
              {"role": m["role"], "content": m["content"]}
              for m in messages
          ],
          stream=True,
          stream_options={
              "include_usage": True
          }
      )

def count_tokens(messages):
    """Count the number of tokens in the messages.
    
    Args:
        messages: List of message objects with role and content
    
    Returns:
        Number of tokens in the messages
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # This is the encoding used by GPT-4
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens

def count_xml_tags(text):
    """Count the number of XML tags in a string.
    
    Args:
        text: String to count XML tags in
        
    Returns:
        Number of XML tags in the string
    """
    # Define the regex pattern for XML tags
    pattern = r'<[^>]+>'

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Return the number of matches
    return len(matches)

def handle_chat_prompt(prompt, page):
    """Process a user prompt, send to Azure OpenAI and display the response.
    
    Args:
        prompt: The user's text input
        st_session_state: Streamlit session state object containing messages
        
    Returns:
        None - updates the session state and UI directly
    """
    # Cleanup prompt
    if count_xml_tags(prompt) > 0:
        logger.debug("Prompt contains XML tags.")
        # embed documents to avoid harm
        prompt = f"<documents>{prompt}</documents>"

    # Echo the user's prompt to the chat window
    page["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    # Calculate tokens in the input
    input_tokens = count_tokens(page["messages"])
    
    # Send the user's prompt to Azure OpenAI and display the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        completion = None
        for response in create_chat_completion(page["messages"]):
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
    page["messages"].append({"role": "assistant", "content": full_response})
    
    # Display token usage
    if completion and completion.usage:
        st.caption(f"""Token usage for this interaction:
        - Input tokens: {input_tokens}
        - Output tokens: {completion.usage.completion_tokens}
        - Total tokens: {completion.usage.total_tokens}""")

def load_prompt_files(persona_file_path, content_file_path):
    """Load content from prompt files and create initial messages.
    
    Args:
        persona_file_path: Path to the persona/system prompt file
        content_file_path: Path to the content/context file
        
    Returns:
        List of message objects with the system prompts loaded
    """
    # Read the specified files into a list of strings
    with open(persona_file_path, "r") as f:
        system_prompt = f.read()
    
    with open(content_file_path, "r") as f:
        system_document = f.read()
    
    # Determine the content label based on the file name
    content_label = "Find the AI Discovery Cards definitions below" if "discovery" in content_file_path else "Find the use case below"
    
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "system",
            "content": f"{content_label}:\n\n{system_document}"
        }
    ]
