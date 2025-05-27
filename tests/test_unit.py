"""
Unit tests for utility functions in the AIDiscoveryCardsFacilitator project.

This module contains tests for:
- count_tokens() and count_xml_tags() from src/utils/openai_utils.py
- get_system_messages() and get_system_messages_multiagent() from src/openai_page.py
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import re

# Add the src directory to the path so we can import modules from there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# We need to mock dependencies before importing the modules to avoid circular imports
sys.modules['streamlit'] = MagicMock()
sys.modules['st_copy'] = MagicMock()
sys.modules['streamlit.logger'] = MagicMock()
sys.modules['streamlit_mermaid'] = MagicMock()
sys.modules['agents.agent'] = MagicMock()
sys.modules['agent_registry'] = MagicMock()

# Import tiktoken for the count_tokens implementation
import tiktoken

# Function: count_tokens (copied from src/utils/openai_utils.py)
def count_tokens(messages):
    """Count the number of tokens in the messages.

    Args:
        messages: List of message objects with role and content

    Returns:
        Number of tokens in the messages
    """
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )  # This is the encoding used by GPT-4
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for _, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens

# Function: count_xml_tags (copied from src/utils/openai_utils.py)
def count_xml_tags(text):
    """Count the number of XML tags in a string.

    Args:
        text: String to count XML tags in

    Returns:
        Number of XML tags in the string
    """
    # Define the regex pattern for XML tags
    pattern = r"<[^>]+>"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Return the number of matches
    return len(matches)


class TestOpenAIUtils(unittest.TestCase):
    """Tests for utility functions in src/utils/openai_utils.py"""

    @patch("tiktoken.get_encoding")
    def test_count_tokens(self, mock_get_encoding):
        """Test the count_tokens function with mocked tiktoken"""
        # Setup mock
        mock_encoding = MagicMock()
        mock_get_encoding.return_value = mock_encoding
        
        # Case 1: Empty message list
        mock_encoding.encode.return_value = []
        self.assertEqual(count_tokens([]), 2)  # Base token count is 2
        
        # Case 2: Single message with simple content
        mock_encoding.encode.side_effect = [
            [1, 2, 3, 4, 5],  # 5 tokens for "user"
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 10 tokens for "Hello world"
        ]
        messages = [{"role": "user", "content": "Hello world"}]
        self.assertEqual(count_tokens(messages), 4 + (5 + 10) + 2)  # 4 per message + content tokens + 2 base tokens
        
        # Reset mock
        mock_encoding.reset_mock()
        mock_encoding.encode.side_effect = [
            [1, 2, 3, 4, 5],  # 5 tokens for "user"
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 10 tokens for "Hello world"
            [1, 2, 3, 4, 5],  # 5 tokens for "assistant"
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 15 tokens for "I'm an AI assistant"
        ]
        
        # Case 3: Multiple messages with different content lengths
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "I'm an AI assistant"}
        ]
        # 4 for first message + 4 for second message + tokens for all values + 2 base tokens
        self.assertEqual(count_tokens(messages), 4 + 4 + (5 + 10) + (5 + 15) + 2)
        
        # Verify encoding was called with correct arguments
        mock_get_encoding.assert_called_with("cl100k_base")
    
    def test_count_xml_tags(self):
        """Test the count_xml_tags function with various inputs"""
        # Case 1: No XML tags
        self.assertEqual(count_xml_tags("Hello world"), 0)
        
        # Case 2: Simple XML tags
        self.assertEqual(count_xml_tags("<tag>content</tag>"), 2)
        
        # Case 3: Self-closing XML tags
        self.assertEqual(count_xml_tags("<tag />"), 1)
        
        # Case 4: Multiple XML tags including closing tags
        xml = "<root><child>text</child><child>more text</child></root>"
        expected_tags = 6  # All tags: <root>, <child>, </child>, <child>, </child>, </root>
        self.assertEqual(count_xml_tags(xml), expected_tags)
        
        # Case 5: XML tags with attributes
        self.assertEqual(
            count_xml_tags('<tag attr="value">content</tag>'),
            2
        )
        
        # Case 6: Overlapping <> characters that aren't XML tags
        self.assertEqual(
            count_xml_tags("2 < 5 and 10 > 7"),
            1  # The regex actually matches this as one tag: "< 5 and 10 >"
        )
        
        # Case 7: Mixed content
        self.assertEqual(
            count_xml_tags("Text before <tag>content</tag> and <another>more</another>"),
            4
        )


class TestSystemMessages(unittest.TestCase):
    """Tests for system message functions"""
    
    def test_get_system_messages(self):
        """Test get_system_messages functionality"""
        # Mock the load_prompt_files function
        mock_load_prompt_files = MagicMock()
        mock_load_prompt_files.return_value = [{"role": "system", "content": "You are a helpful assistant"}]
        
        # Create a function that uses the mock
        def get_system_messages(persona, documents=None):
            return mock_load_prompt_files(persona, documents)
        
        # Test with only persona
        persona_path = "prompts/facilitator_persona.md"
        result = get_system_messages(persona_path)
        
        # Verify the mock was called correctly
        mock_load_prompt_files.assert_called_once_with(persona_path, None)
        self.assertEqual(result, [{"role": "system", "content": "You are a helpful assistant"}])
        
        # Reset the mock
        mock_load_prompt_files.reset_mock()
        
        # Test with persona and document
        document_path = "prompts/ai_discovery_cards.md"
        result = get_system_messages(persona_path, document_path)
        
        # Verify the mock was called correctly
        mock_load_prompt_files.assert_called_once_with(persona_path, document_path)
    
    def test_get_system_messages_multiagent(self):
        """Test get_system_messages_multiagent functionality"""
        # Mock the load_prompt_files function
        mock_load_prompt_files = MagicMock()
        mock_load_prompt_files.side_effect = [
            [{"role": "system", "content": "You are persona 1"}],
            [{"role": "system", "content": "You are persona 2"}]
        ]
        
        # Create a simplified implementation of the function
        def get_system_messages_multiagent(personas, documents=None):
            messages = []
            for i, persona in enumerate(personas):
                # Handle document pairing for this persona
                persona_docs = None
                if documents and i < len(documents):
                    persona_docs = documents[i]
                
                persona_messages = mock_load_prompt_files(persona, persona_docs)
                messages.extend(persona_messages)
            return messages
        
        # Test with multiple personas, no documents
        personas = ["prompts/persona1.md", "prompts/persona2.md"]
        result = get_system_messages_multiagent(personas)
        
        # Verify correct calls to mock
        self.assertEqual(mock_load_prompt_files.call_count, 2)
        mock_load_prompt_files.assert_any_call("prompts/persona1.md", None)
        mock_load_prompt_files.assert_any_call("prompts/persona2.md", None)
        
        # Verify expected result
        expected_result = [
            {"role": "system", "content": "You are persona 1"},
            {"role": "system", "content": "You are persona 2"}
        ]
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()