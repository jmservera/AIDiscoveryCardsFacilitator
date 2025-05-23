"""
Test script for Mermaid diagrams in the chat interface
"""

import streamlit as st
from streamlit_mermaid import st_mermaid

from utils.openai_utils import extract_mermaid_diagrams, render_response_with_mermaid

st.title("Mermaid Test")

# Sample mermaid diagram
mermaid_code = """
graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Debug]
    D --> A
"""

# Display the diagram
st.markdown("### Sample Mermaid Diagram")
st_mermaid(mermaid_code)

# Show the code
st.markdown("### Mermaid Code")
st.code(f"```mermaid\n{mermaid_code}\n```", language="markdown")

# Show how it would look in a response
st.markdown("### How it would look in a response")
st.markdown(
    f"This is some text before the diagram.\n\n```mermaid\n{mermaid_code}\n```\n\nThis is some text after the diagram."
)

# Test the extraction function
st.markdown("### Testing Mermaid Extraction")
test_text = f"""Here is a diagram showing a workflow:

```mermaid
{mermaid_code}
```

And here's some more text after the diagram.

Here's another diagram:

```mermaid
flowchart LR
    A[Hard edge] -->|Link text| B(Round edge)
    B --> C{{Decision}}
    C -->|One| D[Result one]
    C -->|Two| E[Result two]
```
"""

diagrams = extract_mermaid_diagrams(test_text)
st.write(f"Found {len(diagrams)} diagrams")

# Render using our utility function
st.markdown("### Testing Render Response With Mermaid")
st.write("The diagrams should be rendered below:")
render_response_with_mermaid(test_text)
