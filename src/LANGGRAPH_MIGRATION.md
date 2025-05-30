# LangGraph Migration Documentation

## Overview

This document describes the migration from direct Azure OpenAI API calls to LangGraph-based chat workflows in the AI Discovery Cards Facilitator application.

## Key Changes Made

### 1. Dependencies Updated

- Added `langgraph>=0.2.45` for workflow orchestration
- Added `langchain-openai>=0.2.8` for Azure OpenAI integration via LangChain

### 2. New LangGraph Workflow Implementation

**File: `src/workflows/chat_graph.py`**

- Created `ChatGraph` class that replaces direct Azure OpenAI API calls
- Implements LangGraph workflows for chat completion
- Maintains backward compatibility with OpenAI streaming response format
- Supports both synchronous and asynchronous execution
- Includes fallback responses for testing and when Azure credentials are unavailable

### 3. Agent Class Refactoring

**File: `src/agents/agent.py`**

- Replaced direct `openai.AzureOpenAI` client usage with `ChatGraph` workflows
- Removed `get_client()` method as it's no longer needed
- Updated `create_chat_completion()` to use LangGraph execution
- Maintained the same interface for backward compatibility
- Added comprehensive documentation about the migration

### 4. Utility Functions Updated

**File: `src/utils/openai_utils.py`**

- Updated `handle_chat_prompt()` to support both old and new interfaces
- Added backward compatibility wrapper for existing Streamlit UI code
- Fixed circular import issues using TYPE_CHECKING
- Added migration documentation to all relevant functions

## Migration Benefits

1. **Better Workflow Management**: LangGraph provides better orchestration for complex agent workflows
2. **Enhanced Flexibility**: Support for both single and multi-agent conversations through unified workflows
3. **Improved Integration**: Native LangChain integration with Azure OpenAI
4. **Backward Compatibility**: Existing UI code continues to work without changes
5. **Future-Proofing**: Ready for advanced LangGraph features like memory, tools, and complex workflows

## Usage Patterns

### New Pattern (Recommended)
```python
from agents.single_agent import SingleAgent

agent = SingleAgent('my_agent', 'prompts/persona.md', 'gpt-4o')
messages = [{'role': 'user', 'content': 'Hello'}]
response_stream = agent.create_chat_completion(messages)
```

### Old Pattern (Still Supported)
```python
from utils.openai_utils import handle_chat_prompt

# This still works - creates temporary agent internally
handle_chat_prompt(prompt, page, model, temperature)
```

## Testing

The migration includes comprehensive fallback mechanisms:

- When Azure credentials are not available, the system provides mock responses
- All existing interfaces remain functional
- Both single and multi-agent workflows are tested and working
- Streaming responses maintain the same format as the original OpenAI implementation

## Future Enhancements

The LangGraph foundation enables future enhancements such as:

- Memory persistence across conversations
- Tool integration for external API calls
- Complex multi-step workflows
- Advanced agent collaboration patterns
- Custom node types for specialized processing