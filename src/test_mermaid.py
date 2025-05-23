"""
Test script for Mermaid diagrams in the chat interface
"""

import streamlit as st
from streamlit_mermaid import st_mermaid

from utils.openai_utils import (
    extract_mermaid_diagrams,
    get_diagram_scale_factor,
    render_response_with_mermaid,
    set_diagram_scale_factor,
)

st.title("Mermaid Test")

# Add scaling slider in sidebar
st.sidebar.title("Diagram Settings")
scale_factor = st.sidebar.slider(
    "Diagram Size Scale",
    min_value=0.5,
    max_value=2.0,
    value=get_diagram_scale_factor(),
    step=0.1,
    help="Adjust the size of the Mermaid diagrams (1.0 is the default size)",
)
set_diagram_scale_factor(scale_factor)

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
```"""

# Add test for a larger, more complex diagram
complex_diagram_code = """classDiagram
    class Animal {
        +int age
        +String gender
        +isMammal()
        +mate()
    }
    class Duck {
        +String beakColor
        +swim()
        +quack()
    }
    class Fish {
        -int sizeInFeet
        -canEat()
    }
    class Zebra {
        +bool is_wild
        +run()
    }
    Animal <|-- Duck
    Animal <|-- Fish
    Animal <|-- Zebra
    
    class Customer {
        +String name
        +String email
        +String address
        +getDetails()
    }
    class Order {
        +int orderNumber
        +Date orderDate
        +float totalAmount
        +addItem()
        +removeItem()
        +calculateTotal()
    }
    class OrderItem {
        +String productName
        +int quantity
        +float price
        +calculateSubtotal()
    }
    Customer "1" --> "*" Order
    Order "1" *-- "*" OrderItem
```

And more text after the large diagram."""


diagrams = extract_mermaid_diagrams(test_text)
st.write(f"Found {len(diagrams)} diagrams")

# Render using our utility function
st.markdown("### Testing Render Response With Mermaid")
st.write("The diagrams should be rendered below:")
render_response_with_mermaid(test_text)

# Test with complex diagram
st.markdown("### Testing Complex Diagram Auto-Sizing")
st.write("This diagram should adjust its height based on content complexity:")

complex_text = f"""Here's a larger, more complex diagram to test our auto-sizing:

```mermaid
{complex_diagram_code}
```

And more text after the large diagram."""

render_response_with_mermaid(complex_text)

# Test with wide diagram
st.markdown("### Testing Wide Diagram Flex Width")
st.write("This diagram should expand to fill container width:")

wide_diagram_code = """flowchart LR
    A[Module A] -->|Data Flow| B[Module B] -->|Process| C[Module C] -->|Output| D[Module D]
    A -->|Alternative| E[Module E] -->|Process| F[Module F] -->|Join| C
    B -->|Optional| G[Module G] -->|Support| H[Module H] -->|Join| D
    E -->|Alternative| I[Module I] -->|Process| J[Module J] -->|Join| F
    K[External Input] -->|Feed| A
    K -->|Feed| E
    D -->|Feedback| L[Monitoring]
    L -->|Adjust| A"""

wide_text = f"""Here's a wide diagram that should adapt to page width:

```mermaid
{wide_diagram_code}
```

And some text after the wide diagram."""

render_response_with_mermaid(wide_text)

# Test with multiple diagram types
st.markdown("### Testing Multiple Diagram Types")
st.write("This test includes sequence, gantt, and pie charts:")

multi_diagram_text = """Here's a comprehensive test with multiple diagram types:

First, a sequence diagram:

```mermaid
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>John: Hello John, how are you?
    loop Health Check
        John->>John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail!
    John-->>Alice: Great!
    John->>Bob: How about you?
    Bob-->>John: Jolly good!
```

Next, a gantt chart:

```mermaid
gantt
    title A Gantt Diagram
    dateFormat  YYYY-MM-DD
    section Section
    A task           :a1, 2024-05-01, 30d
    Another task     :after a1, 20d
    section Another
    Task in sec      :2024-05-12, 12d
    another task     :24d
```

And finally a pie chart:

```mermaid
pie title Pets Adoption
    "Dogs" : 386
    "Cats" : 285
    "Birds" : 141
    "Fish" : 98
```

This test showcases flexibility with multiple diagram types.
"""
